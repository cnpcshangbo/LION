import numpy as np

class DropLidarBeams:
    def __init__(self, process_cfg, point_cloud_range, training, num_point_features, **kwargs):
        self.name = 'DropLidarBeams' # Required for OpenPCDet
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        
        # Get config parameters from process_cfg which comes from the YAML
        # Provide default values if not specified in YAML
        self.num_assumed_rings = process_cfg.get('NUM_ASSUMED_RINGS', 64)
        self.drop_ratio = process_cfg.get('DROP_RATIO', 0.5) 
        # Example: drop_pattern = 'even' (drop even rings 0, 2, 4...) 
        # or 'odd' (drop odd rings 1, 3, 5...)
        self.drop_pattern = process_cfg.get('DROP_PATTERN', 'even') 

    def __call__(self, data_dict):
        '''
        Args:
            data_dict:
                points: (N, 3 + C_in)
        Returns:
            data_dict:
                points: (N_filtered, 3 + C_in)
        '''
        points = data_dict.get('points', None)
        if points is None:
            return data_dict

        if self.drop_ratio <= 0: # No beams to drop
            return data_dict

        # Calculate vertical angle for each point
        # points[:, 0] is x, points[:, 1] is y, points[:, 2] is z
        vertical_angles = np.arctan2(points[:, 2], np.sqrt(points[:, 0]**2 + points[:, 1]**2))

        # Determine min and max vertical angles from the entire point cloud
        min_angle = np.min(vertical_angles)
        max_angle = np.max(vertical_angles)

        # Discretize angles into num_assumed_rings bins
        # This assigns an estimated ring_id (0 to num_assumed_rings - 1) to each point
        # Ensure that points at max_angle are correctly binned into the last ring.
        ring_ids = np.floor((vertical_angles - min_angle) / (max_angle - min_angle + 1e-6) * self.num_assumed_rings).astype(int)
        ring_ids = np.clip(ring_ids, 0, self.num_assumed_rings - 1) # Ensure IDs are within bounds

        # Select points to keep based on drop_pattern
        if self.drop_pattern == 'even': # Drop even rings (0, 2, ...), keep odd rings (1, 3, ...)
            keep_mask = (ring_ids % 2) != 0
        elif self.drop_pattern == 'odd': # Drop odd rings (1, 3, ...), keep even rings (0, 2, ...)
            keep_mask = (ring_ids % 2) == 0
        else: # Default to keeping all if pattern is not recognized
            keep_mask = np.ones(len(points), dtype=bool)
        
        # For a generic drop_ratio, we could decide to drop a certain percentage of rings.
        # However, the 'even'/'odd' pattern is more aligned with how beams are structured.
        # If drop_ratio is used, it implies dropping a fraction of total rings.
        # For simplicity with 'even'/'odd', drop_ratio=0.5 is implicit.
        # If a different drop_ratio is desired with a more random ring selection,
        # this logic would need to be more complex, e.g., identify unique rings
        # and randomly select a subset to drop.
        # For now, we stick to the even/odd pattern which naturally drops 50%.

        filtered_points = points[keep_mask]
        
        data_dict['points'] = filtered_points
        
        return data_dict
