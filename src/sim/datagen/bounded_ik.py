"""Bounded Newton FK/SciPy pose steps shared by SO101 tasks and SP7.

Kinematics and Jacobians remain Newton-owned.
"""

class BoundedPoseStep:
    """Solve the pose and joint bounds together for the redundant arm chain."""

    def __init__(self, arm, options, kinematics):
        from sim.datagen.ik import _arm_bounds
        self.arm, self.options, self.kinematics = arm, options, kinematics
        self.bounds = _arm_bounds(arm, options)

    def solve(self, q, ee_pos, ee_quat, jacobian, target_pos, target_quat=None):
        import numpy as np
        import torch
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        from sim.datagen.ik import IKResult

        def numpy(value):
            return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

        positions = np.atleast_2d(numpy(ee_pos))
        orientations = np.atleast_2d(numpy(ee_quat))
        targets = np.atleast_2d(numpy(target_pos))
        rotations = None if target_quat is None else np.atleast_2d(numpy(target_quat))
        original = np.atleast_2d(numpy(q)).astype(float)
        result = original.copy()
        ids = np.array(self.bounds['ids'])
        caps = np.array(self.bounds['step_caps'])
        physical_low, physical_high = np.array(self.bounds['lower']), np.array(self.bounds['upper'])
        axes = [i for i, name in enumerate(('x', 'y', 'z')) if name in self.arm.ik.orientation_axes]
        position_errors, orientation_errors, limited = [], [], []
        for env, measured in enumerate(original):
            current = self.kinematics.fk(measured)
            delta = targets[env] - positions[env]
            position_errors.append(float(np.linalg.norm(delta)))
            delta *= min(1., self.options.max_position_step_m/max(np.linalg.norm(delta), 1e-12))
            nominal_target = current[:3, 3] + delta
            desired_rotation = None
            orientation_error = 0.
            if rotations is not None and axes:
                actual_rotation = Rotation.from_quat(orientations[env]).as_matrix()
                requested_rotation = Rotation.from_quat(rotations[env]).as_matrix()
                orientation_error = float(np.linalg.norm(
                    Rotation.from_matrix(requested_rotation @ actual_rotation.T).as_rotvec()[axes]))
                desired_rotation = requested_rotation @ actual_rotation.T @ current[:3, :3]
            orientation_errors.append(orientation_error)
            lower = np.maximum(physical_low, measured[ids]-caps)
            upper = np.minimum(physical_high, measured[ids]+caps)
            if np.any(lower >= upper):
                raise ValueError('Measured joint is outside its physical command window')
            origin = np.clip(measured[ids], lower+1e-9, upper-1e-9)
            middle = (physical_low+physical_high)/2
            avoidance = .01*np.sqrt(self.arm.ik.joint_limit_avoidance_gain)

            def evaluate(values, derivative=False):
                state = measured.copy()
                state[ids] = values
                pose = self.kinematics.fk(state)
                residual = [pose[:3, 3]-nominal_target]
                matrix = self.kinematics.jacobian(state)[:, ids] if derivative else None
                derivatives = [matrix[:3]] if derivative else []
                if desired_rotation is not None:
                    error = Rotation.from_matrix(pose[:3, :3] @ desired_rotation.T).as_rotvec()
                    residual.append(self.options.orientation_weight*error[axes])
                    if derivative:
                        x, y, z = error
                        cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                        angle = np.linalg.norm(error)
                        factor = 1/12 if angle < 1e-4 else (
                            1/angle**2-(1+np.cos(angle))/(2*angle*np.sin(angle)))
                        inverse = np.eye(3)-.5*cross+factor*(cross@cross)
                        derivatives.append(self.options.orientation_weight*(inverse@matrix[3:])[axes])
                residual.extend((self.options.damping*(values-origin), avoidance*(values-middle)))
                if derivative:
                    derivatives.extend((self.options.damping*np.eye(len(ids)), avoidance*np.eye(len(ids))))
                    return np.vstack(derivatives)
                return np.concatenate(residual)

            solution = least_squares(evaluate, origin, jac=lambda v: evaluate(v, True),
                bounds=(lower, upper), max_nfev=12, ftol=1e-5, xtol=1e-5, gtol=1e-7)
            result[env, ids] = solution.x
            limited.append(bool(np.any(np.isclose(solution.x, lower, atol=1e-5)) or
                                np.any(np.isclose(solution.x, upper, atol=1e-5))))
        tensor = lambda values: torch.as_tensor(values, dtype=q.dtype, device=q.device)
        pe, oe = tensor(position_errors), tensor(orientation_errors)
        return IKResult(tensor(result), pe, oe,
            (pe <= self.options.position_tolerance_m) & (oe <= self.options.orientation_tolerance_rad),
            torch.as_tensor(limited, dtype=torch.bool, device=q.device))
