import polars as pl
from scipy.spatial.transform import Rotation
df = pl.read_csv('/srv/STP/data/FAB/_dep/FAB96284M_BuildB_20220921_164620.csv')

quats = df[['Head_quat_x', 'Head_quat_y', 'Head_quat_z', 'Head_quat_w']]
rot = Rotation.from_quat(quats)
rot_euler = rot.as_euler('xyz', degrees=True)
euler_df = pl.DataFrame(rot_euler)
tmp_df = df.with_columns(euler_df['column_0'].alias("Head_euler_x"))
tmp_df = tmp_df.with_columns(euler_df['column_1'].alias("Head_euler_y"))
tmp_df = tmp_df.with_columns(euler_df['column_2'].alias("Head_euler_z"))

quats = df[['LeftHand_quat_x', 'LeftHand_quat_y', 'LeftHand_quat_z', 'LeftHand_quat_w']]
rot = Rotation.from_quat(quats)
rot_euler = rot.as_euler('xyz', degrees=True)
euler_df = pl.DataFrame(rot_euler)
tmp_df = tmp_df.with_columns(euler_df['column_0'].alias("LeftHand_euler_x"))
tmp_df = tmp_df.with_columns(euler_df['column_1'].alias("LeftHand_euler_y"))
tmp_df = tmp_df.with_columns(euler_df['column_2'].alias("LeftHand_euler_z"))

quats = df[['RightHand_quat_x', 'RightHand_quat_y', 'RightHand_quat_z', 'RightHand_quat_w']]
rot = Rotation.from_quat(quats)
rot_euler = rot.as_euler('xyz', degrees=True)
euler_df = pl.DataFrame(rot_euler)
tmp_df = tmp_df.with_columns(euler_df['column_0'].alias("RightHand_euler_x"))
tmp_df = tmp_df.with_columns(euler_df['column_1'].alias("RightHand_euler_y"))
tmp_df = tmp_df.with_columns(euler_df['column_2'].alias("RightHand_euler_z"))

tmp_df.write_csv('/srv/STP/data/FAB/_dep/FAB96284M_BuildB.csv')