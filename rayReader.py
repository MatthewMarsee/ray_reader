import h5py
import numpy as np
from scipy.constants import c


class Rays:
    #Reader class for ray libraries stored in HDF5 format.


    # Initialization
    def __init__(self, filename):
        self.filename = filename
        self.f = h5py.File(filename, "r")

        # Raw datasets
        self._rays = self.f["rays"]
        self._launch_angles_labels = self.f["launch_angle_labels"][:].astype(str)
        self._launch_angles_values = self._launch_angles_labels.astype(np.float32)
        self._valid_mask = self.f["valid_mask"][:].astype(bool)

        # Valid indices
        self._valid_indices = np.where(self._valid_mask)[0]

        if len(self._valid_indices) == 0:
            raise RuntimeError("No valid rays found in file.")

        self._valid_launch_angles = self._launch_angles_values[self._valid_indices]

        # Public size
        self.N = len(self._valid_indices)

    # managing functions
    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    
    # Utilities
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        """Access ray by valid index."""
        if idx < 0 or idx >= self.N:
            raise IndexError("Ray index out of range.")

        real_idx = self._valid_indices[idx]
        return self._rays[real_idx]
    #
    
    @staticmethod
    def _strip_nan_padding(ray):
        """Remove NaN-padded tail from ray array (3, N) and return cleaned copy."""
        ray = np.asarray(ray)

        mask = np.all(np.isfinite(ray), axis=0)

        if not np.any(mask):
            raise RuntimeError("Ray contains no finite samples.")

        return ray[:, mask]

    # Accessors
    def get_time(self, ray, depth):
        """Return propagation time (ns) at closest depth, ignoring NaNs."""
        idx = np.nanargmin(np.abs(ray[1] - depth))
        return float(ray[2][idx])

    def get_r(self, ray, depth):
        """Return radial distance (m) at closest depth, ignoring NaNs."""
        idx = np.nanargmin(np.abs(ray[1] - depth))
        return float(ray[0][idx])
    

    def get_launch_angle(self, idx):
        """Return launch angle for valid index."""
        real_idx = self._valid_indices[idx]
        return float(self._launch_angles_values[real_idx])

    def get_all_launch_angles(self):
        """Return copy of all valid launch angles."""
        return self._valid_launch_angles.copy()

    def get_raw_index(self, idx):
        """Convert valid index to raw file index."""
        return int(self._valid_indices[idx])



    # Zenith Lookup
    @staticmethod
    def source_zenith_to_launch_angle(source_zenith, n_air=1.0, n_surface=1.24, mode="rad"):
        """
        Convert source zenith → launch angle.

        mode:
            "rad" → input/output in radians
            "deg" → input/output in degrees
        """

        if mode not in ("rad", "deg"):
            raise ValueError("mode must be 'rad' or 'deg'")

        # convert input to radians
        if mode == "deg":
            source_zenith = np.deg2rad(source_zenith)

        refracted = np.arcsin(n_air / n_surface * np.sin(source_zenith))
        launch_angle = np.pi - refracted

        if mode == "deg":
            launch_angle = np.rad2deg(launch_angle)

        return launch_angle  


    def get_ray_from_source_zenith(self, source_zenith, n_air=1.0, n_surface=1.24, mode="rad", return_index=False):
        """
        Find nearest ray corresponding to a given source zenith.

        mode:
            "rad" → input/output in radians
            "deg" → input/output in degrees
        """

        if mode not in ("rad", "deg"):
            raise ValueError("mode must be 'rad' or 'deg'")

        # ---- convert input → radians if needed ----
        if mode == "deg":
            source_zenith_rad = np.deg2rad(source_zenith)
        else:
            source_zenith_rad = source_zenith

        #internal calcularions in rad
        launch_angle = self.source_zenith_to_launch_angle(source_zenith_rad,n_air=n_air,n_surface=n_surface,mode="rad")

        # ---- nearest ray lookup ----
        #dict keys ('valid_launch_angles') are in degrees for readaiblity
        idx_valid = np.argmin(np.abs(self._valid_launch_angles - np.rad2deg(launch_angle))) 
        real_idx = self._valid_indices[idx_valid]

        ray = self._strip_nan_padding(self._rays[real_idx])

        #'launch_angles_values' come from dict, are in degrees
        if mode == "deg":
            matched_angle = float(self._launch_angles_values[real_idx])
        else:
            matched_angle = float(np.deg2rad(self._launch_angles_values[real_idx]))  

        if return_index:
            return ray, matched_angle, int(idx_valid)

        return ray, matched_angle



    # Summary
    def summary(self):
        print("Ray file:", self.filename)
        print("Total rays (raw):", len(self._rays))
        print("Valid rays:", self.N)
        print("Launch angle range:",
              f"{self._valid_launch_angles.min():.3f} - "
              f"{self._valid_launch_angles.max():.3f} deg")
        
    

    # ============================================================
    # Time Delay (Plane Wave + Ray Tracer)
    def time_delay_from_source(self, chA_pos, chB_pos, source_pos):

        chA_pos = np.asarray(chA_pos, dtype=float)
        chB_pos = np.asarray(chB_pos, dtype=float)
        source_pos = np.asarray(source_pos, dtype=float)

        # Enforce 3D positions
        if chA_pos.shape != (3,) or chB_pos.shape != (3,):
            raise ValueError("Both chA_pos and chB_pos must be 3D vectors [x,y,z] for plane wave.")

        source_zenith = source_pos[0]
        source_azimuth = source_pos[1]

        # Plane-wave baseline projection
        baseline_EN = np.array([ chB_pos[0] - chA_pos[0], chB_pos[1] - chA_pos[1] ])
        src_vec_EN = -np.array([ np.cos(source_azimuth), np.sin(source_azimuth) ])
        proj_baseline = np.dot(src_vec_EN, baseline_EN)

        # Load ray for this zenith
        ray, _ = self.get_ray_from_source_zenith(np.rad2deg(source_zenith))
        ray_r, ray_z, ray_t = ray

        # Ice radial offsets
        r_ice_A = np.interp(chA_pos[2], ray_z[::-1], -ray_r[::-1])
        r_ice_B = np.interp(chB_pos[2], ray_z[::-1], -ray_r[::-1])

        ray_entry_shift = proj_baseline + (r_ice_B - r_ice_A)

        # Air propagation correction
        t_air_A = ray_entry_shift * np.sin(source_zenith) / c
        t_air_A *= 1e9  # convert to ns

        t_air_B = 0.0

        # Ice propagation times
        t_ice_A = np.interp(chA_pos[2], ray_z[::-1], ray_t[::-1])
        t_ice_B = np.interp(chB_pos[2], ray_z[::-1], ray_t[::-1])


        return (t_ice_B - t_air_B) - (t_ice_A - t_air_A)
    #

    def time_delay_from_ray(self,selected_ray, chA_pos, chB_pos, source_pos):

        chA_pos = np.asarray(chA_pos, dtype=float)
        chB_pos = np.asarray(chB_pos, dtype=float)
        source_pos = np.asarray(source_pos, dtype=float)#SHOULD BE IN RADIANS

        # Enforce 3D positions
        if chA_pos.shape != (3,) or chB_pos.shape != (3,):
            raise ValueError("Both chA_pos and chB_pos must be 3D vectors [x,y,z] for plane wave.")

        source_zenith = source_pos[0]
        source_azimuth = source_pos[1]

        # Plane-wave baseline projection
        baseline_EN = np.array([chB_pos[0] - chA_pos[0], chB_pos[1] - chA_pos[1]])
        src_vec_EN = -np.array([np.cos(source_azimuth), np.sin(source_azimuth)])
        proj_baseline = np.dot(src_vec_EN, baseline_EN)

        # Load ray for this zenith
        ray_r, ray_z, ray_t = selected_ray

        # Remove NaN-padded tail
        ray_r, ray_z, ray_t = self._strip_nan_padding(selected_ray)

        # Ice radial offsets
        r_ice_A = np.interp(chA_pos[2], ray_z[::-1], -ray_r[::-1])
        r_ice_B = np.interp(chB_pos[2], ray_z[::-1], -ray_r[::-1])
        ray_entry_shift = proj_baseline + (r_ice_B - r_ice_A)

        # Air propagation correction
        t_air_A = ray_entry_shift * np.sin(source_zenith) / c
        t_air_A *= 1e9  # convert to ns
        t_air_B = 0.0

        # Ice propagation times
        t_ice_A = np.interp(chA_pos[2], ray_z[::-1], ray_t[::-1])
        t_ice_B = np.interp(chB_pos[2], ray_z[::-1], ray_t[::-1])

        # Final delay
        return (t_ice_B - t_air_B) - (t_ice_A - t_air_A)
    
    