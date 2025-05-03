def translate_values(values):
    feature_name = values[0]
    effect_on_A = values[1]
    effect_on_B = values[2]

    stat = feature_name.split("_")[-1]
    if stat in ["mean", "median", "sum", "min", "max", "slope"]:
        effect_on_A = "Long" if effect_on_A > 0 else "Short"
        effect_on_B = "Long" if effect_on_B > 0 else "Short"
    elif stat in ["range", "std", "var", "iqr", "coeff_of_var"]:
        # More variable is better > 0 else Consistent is better
        # for coeff_of_var, this is with more variable or consistent relative to average length of pause
        effect_on_A = "Variable" if effect_on_A > 0 else "Consistent"
        effect_on_B = "Variable" if effect_on_B > 0 else "Consistent"
    elif stat in ["skewness"]:
        # Either mostly shorter pauses with a few long or more consistent pauses is better > 0 else Either mostly longer pauses with a few short or more consistent pauses is better
        effect_on_A = "MostlyShortOrConsistent" if effect_on_A > 0 else "MostlyLongOrConsistent"
        effect_on_B = "MostlyShortOrConsistent" if effect_on_B > 0 else "MostlyLongOrConsistent"
    elif stat in ["kurtosis"]:
        # More extreme (both short and long) is better > 0 else More conistent less outliers better
        effect_on_A = "MoreExtreme" if effect_on_A > 0 else "ConsistentAndLessExtreme"
        effect_on_B = "MoreExtreme" if effect_on_B > 0 else "ConsistentAndLessExtreme"
    return (feature_name, effect_on_A, effect_on_B)

mads = ["0.5", "0.75", "_1_", "1.25"]
steps = ["91", "181", "271", "361", "451", "541"]
feature_set = {}
for mad in mads:
    for step in steps:
        features = []
        for sublist in data:
            sl = [translate_values(corrs.iloc[feature_map[i]].to_list()[:3]) for i in sublist if (step in i) and (mad in i)]
            features.append(sl)
        feature_set[f"{mad}_{step}"] = features