def print_times():
    kvs = []

    files = sorted(os.listdir("/srv/STP/data/FAB/Tracking_A/"))
    for file in files:
        tpl = []
        tpl.append(f'{file.split("_")[0]}_{file.split("_")[1]}')
        df = pl.read_csv(f"/srv/STP/data/FAB/Tracking_A/{file}")
        rh_idx = df['RightHand_TriggerClick_value'].index_of(1)
        if rh_idx != None:
            tpl.append(df['Timestamp'][rh_idx])
        else:
            tpl.append(None)
        
        lh_idx = df['LeftHand_TriggerClick_value'].index_of(1)
        if lh_idx != None:
            tpl.append(df['Timestamp'][lh_idx])
        else:
            tpl.append(None)
        kvs.append(tpl)

    files = sorted(os.listdir("/srv/STP/data/FAB/Tracking_B/"))
    for file in files:
        tpl = []
        tpl.append(f'{file.split("_")[0]}_{file.split("_")[1]}')
        df = pl.read_csv(f"/srv/STP/data/FAB/Tracking_B/{file}")
        rh_idx = df['RightHand_TriggerClick_value'].index_of(1)
        if rh_idx != None:
            tpl.append(df['Timestamp'][rh_idx])
        else:
            tpl.append(None)
        
        lh_idx = df['LeftHand_TriggerClick_value'].index_of(1)
        if lh_idx != None:
            tpl.append(df['Timestamp'][lh_idx])
        else:
            tpl.append(None)
        kvs.append(tpl)

    kvs = sorted(kvs, key=lambda x: x[0])
    for kv in kvs:
        print(kv)

print_times()