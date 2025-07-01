def parse_combo(combo_str):
    """
    Parse a device-motion combination string into separate device and motion tuples.
    
    Args:
        combo_str (str): String in format 'Device1_Device2_motion1_motion2'
        
    Returns:
        tuple: (devices_tuple, motions_tuple)
    """
    # Split the string by underscores
    parts = combo_str.split('_')
    
    # Known devices and motions
    devices = {'LeftHand', 'RightHand', 'Head'}
    motions = {'linvel', 'linacc', 'angvel', 'angacc'}
    
    # Separate devices and motions
    device_parts = []
    motion_parts = []
    
    for part in parts:
        if part in devices:
            device_parts.append(part)
        elif part in motions:
            motion_parts.append(part)
    
    return (tuple(device_parts), tuple(motion_parts))

# Example usage
if __name__ == "__main__":
    # Test with your example
    test_combo = "RightHand_LeftHand_linvel_linacc_angacc"
    devices, motions = parse_combo(test_combo)
    print(f"Devices: {devices}")
    print(f"Motions: {motions}")
    
    # Test with another example
    test_combo2 = "Head_RightHand_LeftHand_linvel_linacc_angvel_angacc"
    devices2, motions2 = parse_combo(test_combo2)
    print(f"\nDevices: {devices2}")
    print(f"Motions: {motions2}") 