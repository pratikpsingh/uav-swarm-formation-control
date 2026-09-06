"""Bridge to one immutable revision of gym-pybullet-drones."""

import json
import warnings
from collections.abc import Callable, Mapping
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from uav_swarm_control._arrays import Float32Array
from uav_swarm_control.configuration import PyBulletSimulatorConfig
from uav_swarm_control.environments.drone_backend import (
    RigidBodyState,
    SimulatorMetadataValue,
    immutable_simulator_metadata,
)

SIMULATOR_PACKAGE = "gym-pybullet-drones"
SIMULATOR_VERSION = "2.2.0"
SIMULATOR_REVISION = "24fccff4da746badb6eaa9f5055cd5d9b6613160"

type Float64Array = NDArray[np.float64]


class _Aviary(Protocol):
    CTRL_TIMESTEP: float
    INIT_XYZS: Float64Array

    def reset(self, *, seed: int) -> tuple[object, object]: ...

    def step(self, action: Float64Array) -> tuple[object, object, object, object, object]: ...

    def close(self) -> None: ...


class _PIDController(Protocol):
    def reset(self) -> None: ...

    def computeControl(
        self,
        *,
        control_timestep: float,
        cur_pos: Float32Array,
        cur_quat: Float32Array,
        cur_vel: Float32Array,
        cur_ang_vel: Float32Array,
        target_pos: Float32Array,
        target_rpy: Float64Array,
        target_vel: Float32Array,
    ) -> tuple[object, object, object]: ...


def _installed_package_details() -> tuple[str, str]:
    try:
        installed = distribution(SIMULATOR_PACKAGE)
    except PackageNotFoundError as error:
        raise RuntimeError(
            "gym-pybullet-drones is not installed; run 'uv sync' from the project root."
        ) from error
    raw_direct_url = installed.read_text("direct_url.json")
    if raw_direct_url is None:
        raise RuntimeError("simulator installation does not contain Git provenance metadata.")
    try:
        direct_url = cast(dict[str, object], json.loads(raw_direct_url))
        vcs_info = cast(dict[str, object], direct_url["vcs_info"])
        revision = cast(str, vcs_info["commit_id"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("simulator Git provenance metadata is malformed.") from error
    if installed.version != SIMULATOR_VERSION or revision != SIMULATOR_REVISION:
        raise RuntimeError(
            "installed simulator does not match the audited version and revision in the adapter."
        )
    return installed.version, revision


def _load_factory(module_name: str, attribute: str) -> Callable[..., object]:
    module = import_module(module_name)
    return cast(Callable[..., object], getattr(module, attribute))


class GymPyBulletDronesBackend:
    """Translate desired world velocities into PID motor commands and physics steps."""

    def __init__(
        self,
        config: PyBulletSimulatorConfig,
        num_drones: int,
        initial_positions: Float32Array,
    ) -> None:
        version, revision = _installed_package_details()
        drone_model_factory = cast(
            Callable[[str], object],
            _load_factory("gym_pybullet_drones.utils.enums", "DroneModel"),
        )
        physics_factory = cast(
            Callable[[str], object],
            _load_factory("gym_pybullet_drones.utils.enums", "Physics"),
        )
        drone_model = drone_model_factory(config.drone_model.value)
        physics = physics_factory(config.physics.value)
        aviary_factory = _load_factory(
            "gym_pybullet_drones.envs.CtrlAviary",
            "CtrlAviary",
        )
        controller_factory = _load_factory(
            "gym_pybullet_drones.control.DSLPIDControl",
            "DSLPIDControl",
        )
        with warnings.catch_warnings():
            # The pinned simulator constructs Gymnasium Box bounds as float64 before
            # explicitly requesting float32. The audited cast is expected and harmless.
            warnings.filterwarnings(
                "ignore",
                message=r".*Box (low|high)'s precision lowered by casting to float32.*",
                module=r"gymnasium\.spaces\.box",
            )
            aviary = aviary_factory(
                drone_model=drone_model,
                num_drones=num_drones,
                neighbourhood_radius=np.inf,
                initial_xyzs=np.asarray(initial_positions, dtype=np.float64),
                initial_rpys=np.zeros((num_drones, 3), dtype=np.float64),
                physics=physics,
                pyb_freq=config.physics_frequency_hz,
                ctrl_freq=config.control_frequency_hz,
                gui=config.gui,
                record=config.record_video,
                obstacles=False,
                user_debug_gui=False,
                output_folder="artifacts/recordings",
            )
        self._aviary = cast(_Aviary, aviary)
        self._controllers = tuple(
            cast(_PIDController, controller_factory(drone_model=drone_model))
            for _ in range(num_drones)
        )
        self._num_drones = num_drones
        self._current_state: RigidBodyState | None = None
        self._closed = False

        get_api_version = cast(Callable[[], int], _load_factory("pybullet", "getAPIVersion"))
        self._metadata = immutable_simulator_metadata(
            {
                "simulator_package": SIMULATOR_PACKAGE,
                "simulator_version": version,
                "simulator_revision": revision,
                "pybullet_api_version": get_api_version(),
                "drone_model": config.drone_model.value,
                "physics": config.physics.value,
                "physics_frequency_hz": config.physics_frequency_hz,
                "control_frequency_hz": config.control_frequency_hz,
                "physics_steps_per_control": config.physics_steps_per_control,
                "gui": config.gui,
                "record_video": config.record_video,
            }
        )

    @property
    def metadata(self) -> Mapping[str, SimulatorMetadataValue]:
        return self._metadata

    def reset(self, initial_positions: Float32Array, *, seed: int) -> RigidBodyState:
        self._ensure_open()
        positions = np.asarray(initial_positions, dtype=np.float64)
        if positions.shape != (self._num_drones, 3):
            raise ValueError(
                f"initial_positions must have shape ({self._num_drones}, 3); "
                f"received {positions.shape}."
            )
        self._aviary.INIT_XYZS = positions.copy()
        observation, _ = self._aviary.reset(seed=seed)
        for controller in self._controllers:
            controller.reset()
        return self._state_from_observation(observation)

    def step_velocity(self, target_velocities_mps: Float32Array) -> RigidBodyState:
        self._ensure_open()
        target_velocities = np.asarray(target_velocities_mps, dtype=np.float32)
        if target_velocities.shape != (self._num_drones, 3):
            raise ValueError(
                f"target_velocities_mps must have shape ({self._num_drones}, 3); "
                f"received {target_velocities.shape}."
            )
        state = self._current_state
        if state is None:
            raise RuntimeError("simulator backend must be reset before stepping.")
        motor_rpms = np.empty((self._num_drones, 4), dtype=np.float64)
        for index, controller in enumerate(self._controllers):
            target_rpy = np.array(
                [0.0, 0.0, state.euler_angles_radians[index, 2]],
                dtype=np.float64,
            )
            rpm, _, _ = controller.computeControl(
                control_timestep=self._aviary.CTRL_TIMESTEP,
                cur_pos=state.positions[index],
                cur_quat=state.quaternions_xyzw[index],
                cur_vel=state.linear_velocities_mps[index],
                cur_ang_vel=state.angular_velocities_rad_s[index],
                target_pos=state.positions[index],
                target_rpy=target_rpy,
                target_vel=target_velocities[index],
            )
            motor_rpms[index] = np.asarray(rpm, dtype=np.float64)
        observation, _, _, _, _ = self._aviary.step(motor_rpms)
        return self._state_from_observation(observation)

    def close(self) -> None:
        if not self._closed:
            self._aviary.close()
            self._closed = True

    def _state_from_observation(self, observation: object) -> RigidBodyState:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (self._num_drones, 20) or not np.all(np.isfinite(values)):
            raise RuntimeError(
                "gym-pybullet-drones must return a finite observation with shape "
                f"({self._num_drones}, 20); received {values.shape}."
            )
        state = RigidBodyState(
            positions=values[:, 0:3],
            quaternions_xyzw=values[:, 3:7],
            euler_angles_radians=values[:, 7:10],
            linear_velocities_mps=values[:, 10:13],
            angular_velocities_rad_s=values[:, 13:16],
            motor_rpms=values[:, 16:20],
        )
        self._current_state = state
        return state

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("cannot use a closed simulator backend.")
