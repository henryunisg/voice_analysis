import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm
from itertools import product
from scipy.stats import chi2

EPS = 1e-6


# ───────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────
def col_for(f, ivl, suf):
    lo, hi = ivl
    return f"{f}_[{lo:.2f}, {hi:.2f})_{suf}"


def xform(s, kind):
    if kind == "log":  return np.log(s + 1 + EPS)
    if kind == "sqrt": return np.sqrt(s + EPS)
    return s  # identity


def prep_matrix(base, extra=None, scaler=None):
    """
    Build the design matrix:
    • `base`     : controls + event dummies  – impute only
    • `extra`    : the feature(s) being tested – impute + RobustScaler
    Returns a DataFrame with an added intercept column.
    """

    # ---------- 1.  split base into numeric / categorical -------------
    num_base = base.select_dtypes(np.number)
    cat_base = base.drop(columns=num_base.columns)

    # median-impute numeric controls (no scaling)
    if len(num_base.columns) > 0:
        num_base_imp = pd.DataFrame(
            SimpleImputer(strategy="median").fit_transform(num_base),
            index=num_base.index,
            columns=num_base.columns,
        )
    else:
        num_base_imp = num_base

    # ---------- 2.  handle extra feature columns ----------------------
    if extra is not None:
        num_extra = extra.select_dtypes(np.number)  # should be all numeric
        cat_extra = extra.drop(columns=num_extra.columns)  # usually empty

        # impute + Robust-scale
        if len(num_extra.columns) > 0:
            num_extra_imp = pd.DataFrame(
                SimpleImputer(strategy="median").fit_transform(num_extra),
                index=num_extra.index,
                columns=num_extra.columns,
            )
            num_extra_scaled = pd.DataFrame(
                scaler().fit_transform(num_extra_imp),
                index=num_extra_imp.index,
                columns=num_extra_imp.columns,
            )
        else:
            num_extra_scaled = num_extra

        # ---------- 3.  concatenate everything ---------------------------
        X = pd.concat(
            [num_base_imp, cat_base, num_extra_scaled, cat_extra], axis=1
        )
    else:
        X = pd.concat(
            [num_base_imp, cat_base], axis=1
        )

    if 'const' not in X:
        X = pd.concat([pd.DataFrame({'const': 1.0}, index=X.index), X], axis=1)
    return X


# ───────────────────────────────────────────────────────────────────────────
# VIF CALCULATION
# ───────────────────────────────────────────────────────────────────────────
def calculate_vif(X, verbose=False):
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    mask = ~X.isna()

    vif_data["feature"] = X[mask].columns
    vif_data["VIF"] = [variance_inflation_factor(X[mask].values, i) for i in range(X[mask].shape[1])]

    # Display the VIFs
    if verbose:
        print(vif_data)
    return vif_data


def glm_linear(y, X, w):
    if w is None:
        res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    else:
        res = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w).fit()

    return {
        "res": res,
        "p": {
            "x": float(res.pvalues.get("x", np.nan)),
            "x_mod": float(res.pvalues.get("x_mod", np.nan)) if "x_mod" in X.columns else np.nan
        },
        "beta": {
            "x": float(res.params.get("x", np.nan)),
            "x_mod": float(res.params.get("x_mod", np.nan)) if "x_mod" in X.columns else np.nan
        },
        "r2": float(res.pseudo_rsquared())
    }


def glm_quadratic(y, X, w):
    if w is None:
        res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    else:
        res = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w).fit()

    return {
        "res": res,
        "p": {
            "x": float(res.pvalues.get("x", np.nan)),
            "x2": float(res.pvalues.get("x2", np.nan)),
            "x_mod": float(res.pvalues.get("x_mod", np.nan)) if "x_mod" in X.columns else np.nan,
            "x2_mod": float(res.pvalues.get("x2_mod", np.nan)) if "x2_mod" in X.columns else np.nan
        },
        "beta": {
            "x": float(res.params.get("x", np.nan)),
            "x2": float(res.params.get("x2", np.nan)),
            "x_mod": float(res.params.get("x_mod", np.nan)) if "x_mod" in X.columns else np.nan,
            "x2_mod": float(res.params.get("x2_mod", np.nan)) if "x2_mod" in X.columns else np.nan
        },
        "r2": float(res.pseudo_rsquared())
    }


def evaluate_feature_combination(
        df, base, cname, ivl, tr, scaler, mod, fn, use_quad, use_weights, target, weights, p_threshold):
    raw = df[cname]
    x = xform(raw, tr).rename("x")
    x2 = (x ** 2).rename("x2")
    y = df[target]
    wts = df[weights]

    extra_cols = pd.DataFrame({'x': x})
    if mod:
        # Handle multiple moderators
        if isinstance(mod, (list, tuple)):
            for m in mod:
                extra_cols[m] = df[m]
        else:
            extra_cols['mod'] = df[mod]

    # Prepare linear design matrix
    X_lin = prep_matrix(base, extra=extra_cols, scaler=scaler)
    if mod:
        # Handle multiple moderators
        if isinstance(mod, (list, tuple)):
            for m in mod:
                X_lin[f'x_{m}'] = X_lin['x'] * X_lin[m]
        else:
            X_lin['x_mod'] = X_lin['x'] * X_lin['mod']

    # Prepare quadratic design matrix
    X_quad = X_lin.copy()
    X_quad['x2'] = X_lin['x'] ** 2
    if mod:
        # Handle multiple moderators
        if isinstance(mod, (list, tuple)):
            for m in mod:
                X_quad[f'x2_{m}'] = X_quad['x2'] * X_quad[m]
        else:
            X_quad['x2_mod'] = X_quad['x2'] * X_quad['mod']

    mask = (~y.isna()) & (~x.isna())
    X = X_quad if use_quad else X_lin
    X = X.loc[mask]
    y_sub = y[mask]
    w = wts[mask] if use_weights else None

    # VIF
    vif_data = calculate_vif(X, verbose=False)
    vif_x = vif_data[vif_data['feature'] == 'x']['VIF'].values[0]

    # Fit model
    res_info = fn(y_sub, X, w)

    # Decide which p-value to use
    if mod is not None:
        if use_quad:
            if isinstance(mod, (list, tuple)):
                p_values = [float(res_info["p"].get(f"x_{m}", np.nan)) for m in mod]
                p_values.extend([float(res_info["p"].get(f"x2_{m}", np.nan)) for m in mod])
                p_value = min(p_values)
            else:
                p_value = min(float(res_info["p"].get("x_mod", np.nan)), float(res_info["p"].get("x2_mod", np.nan)))
        else:
            if isinstance(mod, (list, tuple)):
                p_values = [float(res_info["p"].get(f"x_{m}", np.nan)) for m in mod]
                p_value = min(p_values)
            else:
                p_value = float(res_info["p"].get("x_mod", np.nan))
    else:
        p_value = float(res_info["p"].get("x2" if use_quad else "x", np.nan))

    if p_value <= p_threshold:
        return res_info['res'], dict(
            fn=fn.__name__,
            name=cname,
            feature=cname.split('_[', 1)[0],
            interval=ivl,
            suffix=cname.split("_")[-1],
            trans=tr,
            scaler=scaler.__name__,
            moderator=mod,
            use_quad=use_quad,
            use_weights=use_weights,
            p_value=p_value,
            p_values=res_info["p"],
            beta=res_info["beta"],
            r2=float(res_info["r2"]),
            vif=float(vif_x)
        )
    return res_info['res'], {}


def prepare_df(df, TARGET, CONTROLS, EVENT_COL, SOURCE_COL):
    ### Clean the data
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    # Drop rows with NaN values in the target variable
    df = df.dropna(subset=[TARGET])
    # Fill remaining NaN values with median for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    '''
    x.sum(): This calculates the number of positive cases (n_pos) in each group. In other words, it counts how many 1s are in the TARGET column.
    len(x): This gives the total number of instances in each group (i.e., the total size of the group).
    cw = 0.5 * len(x) / x.sum(): This formula calculates the class weight by balancing the number of positive cases against the total number of cases in each group. It's used to adjust the model for class imbalances in the data.
    '''
    df["cw"] = df.groupby(SOURCE_COL)[TARGET].transform(
        lambda x: 0.5 * len(x) / x.sum() if x.sum() != 0 else 0  # Use n_pos (x.sum()) to calculate cw
    )

    if EVENT_COL:
        event_dummies = pd.get_dummies(df[EVENT_COL], prefix="EVENT", drop_first=True, dtype=int)
        base = pd.concat([df[CONTROLS], event_dummies], axis=1)
    else:
        base = df[CONTROLS]

    if SOURCE_COL:
        source_dummies = pd.get_dummies(df[SOURCE_COL], prefix="source", drop_first=True, dtype=int)
        base = pd.concat([base, source_dummies], axis=1)

    base = prep_matrix(base, extra=None, scaler=None)

    return df, base


def calculate_significant_results():
    # ───────────────────────────────────────────────────────────────────────────
    # CONFIGURATION
    # ───────────────────────────────────────────────────────────────────────────
    CSV_FILE = "robustness_test.csv"
    TARGET = "selected"
    EVENT_COL = None  # "event"
    SOURCE_COL = "source"
    CONTROLS = ['age', 'gender', 'race', 'dress_formality', 'attractiveness']
    MODERATORS = ['credibility', 'emotional_appeal', 'logical_coherence', None]  # None for no moderation
    P_TARGET = 0.05
    INTERVALS = [(0, 1)]  # [(0, 1), (0, .33), (.33, .66), (.66, 1)]
    TRANSFORMS = ["identity", "log"]  # ["identity", "log", "sqrt"]
    FEATURES = ['mean_pitch', 'std_pitch', 'mean_intensity', 'std_intensity', 'duration', 'pause_frequency',
                'silence_ratio', 'speech_to_pause_ratio', 'articulation_rate', 'speech_rate']
    SUFFIXES = ['min', 'max', 'mean', 'std', 'dn_mean', 'dn_std']
    SCALERS = [RobustScaler, StandardScaler]

    MODEL_SPECS = [
        ("glm_linear", glm_linear, False, True),
        ("glm_linear", glm_linear, False, False),
        ("glm_quadratic", glm_quadratic, True, True),
        ("glm_quadratic", glm_quadratic, True, False)
    ]
    # ───────────────────────────────────────────────────────────────────────────
    # LOAD + CLASS-BALANCE WEIGHTS
    # ───────────────────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_FILE)
    df, base = prepare_df(df, TARGET, CONTROLS, EVENT_COL, SOURCE_COL)


    # ───────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ───────────────────────────────────────────────────────────────────────────
    rows = []

    # Calculate VIF for the control and event variables (base)
    vif_data = calculate_vif(base, verbose=True)

    # Pre-calculate all combinations
    combinations = list(product(FEATURES, INTERVALS, SUFFIXES, TRANSFORMS, SCALERS, MODERATORS, MODEL_SPECS))
    for feat, ivl, suf, tr, scaler, mod, (name, fn, use_quad, use_weights) in tqdm(combinations,
                                                                                   desc="Evaluating Features",
                                                                                   unit="combo"):
        cname = col_for(feat, ivl, suf)
        if cname not in df:
            print(f'cname {cname} not found!')
            continue

        result, row = evaluate_feature_combination(
            df=df,
            base=base,
            cname=cname,
            ivl=ivl,
            tr=tr,
            scaler=scaler,
            mod=mod,
            fn=fn,
            use_quad=use_quad,
            use_weights=use_weights,
            target=TARGET,
            weights="cw",
            p_threshold=P_TARGET,
        )
        if row:
            rows.append(row)
    # ───────────────────────────────────────────────────────────────────────────
    # RESULTS
    # ───────────────────────────────────────────────────────────────────────────
    sig_df = (pd.DataFrame(rows)
              .sort_values(["p_value", "fn", "feature"])
              .reset_index(drop=True))

    # sig_df.to_json("data/significant_results.json", orient="records", indent=2)
    sig_df.to_csv("ROBUSTNESS_significant_results.csv", index=False)
    print(f"{len(sig_df)} significant terms @ p ≤ {P_TARGET}")
    print("Saved to: ROBUSTNESS_significant_results.csv")


def calculate_sample(model_type="full"):
    CSV_FILE = "FINAL_voice_analysis.csv"
    TARGET = "selected"
    SOURCE_COL = "source"
    CONTROLS = [] # 'age', 'gender', 'race', 'dress_formality', 'attractiveness'
    EVENT_COL = None

    df = pd.read_csv(CSV_FILE)
    df, base = prepare_df(df, TARGET, CONTROLS, EVENT_COL, SOURCE_COL)
    y = df[TARGET]
    w = df['cw'] if 'cw' in df else None

    if model_type == "full":
        cname = 'mean_pitch_[0.00, 1.00)_dn_mean'  # Example column name from used CSV-File
        ivl = (0, 1)  # Example interval
        tr = "identity"  # Example transformation (identity or log)
        scaler = RobustScaler  # Example scaler (use StandardScaler or RobustScaler)
        mod = None  # Example moderator variable, use None if no moderator
        fn = glm_linear  # Example function (glm_linear or glm_quadratic)
        use_quad = False  # Whether to use quadratic terms (True/False)
        use_weights = True  # Whether to use class weights in the model (True/False)
        target = 'selected'  # Example target column name
        weights = 'cw'  # Example weights column name
        p_threshold = 0.05  # p-value threshold for significance

        # Call the function with the specified arguments
        result, row = evaluate_feature_combination(
            df=df,
            base=base,
            cname=cname,
            ivl=ivl,
            tr=tr,
            scaler=scaler,
            mod=mod,
            fn=fn,
            use_quad=use_quad,
            use_weights=use_weights,
            target=target,
            weights=weights,
            p_threshold=p_threshold,
        )
    elif model_type == "controls":
        # Fit model with only control variables
        if w is None:
            res = sm.GLM(y, base, family=sm.families.Binomial()).fit()
        else:
            res = sm.GLM(y, base, family=sm.families.Binomial(), freq_weights=w).fit()
        result = res
    elif model_type == "null":
        # Fit null/intercept-only model
        X_null = base[["const"]] if "const" in base else pd.DataFrame({"const": 1.0}, index=base.index)
        if w is None:
            res = sm.GLM(y, X_null, family=sm.families.Binomial()).fit()
        else:
            res = sm.GLM(y, X_null, family=sm.families.Binomial(), freq_weights=w).fit()
        result = res
    else:
        raise ValueError("model_type must be 'full', 'controls', or 'null'")

    return result


def print_glm_stats(full_model, null_model):
    print('Log-likelihood:', full_model.llf)
    lr_stat = 2 * (full_model.llf - null_model.llf)
    df_diff = full_model.df_model - null_model.df_model
    lr_pvalue = chi2.sf(lr_stat, df_diff)
    print('LR χ²:', lr_stat)
    print('LR χ² p-value:', lr_pvalue)
    print('McFadden pseudo-R²:', full_model.pseudo_rsquared())


def get_descriptive_statistics(model_type="full"):
    """
    Calculate and print descriptive statistics for the specified model.
    Uses original, unscaled values and excludes interaction terms.
    Includes Selection ('selected') variable to show pitch success rates and correlations.

    Args:
        model_type (str): Type of model to analyze ('full', 'controls', or 'null')
    """
    CSV_FILE = "robustness_test_outliers.csv"
    TARGET = "selected"
    SOURCE_COL = "source"
    CONTROLS = ['age', 'gender', 'race', 'dress_formality', 'attractiveness']
    EVENT_COL = None

    # Load and prepare data
    df = pd.read_csv(CSV_FILE)
    df, base = prepare_df(df, TARGET, CONTROLS, EVENT_COL, SOURCE_COL)

    # Get the relevant data based on model type
    if model_type == "full":
        # Use the same example column as in calculate_sample
        cname = 'mean_intensity_[0.00, 1.00)_dn_mean'
        raw = df[cname]
        x = xform(raw, "identity").rename("x")

        # Create a DataFrame with original values for descriptive statistics
        desc_data = pd.DataFrame()
        desc_data[TARGET] = df[TARGET]
        desc_data['x'] = raw  # Use raw values instead of transformed
        # Add moderators
        mods = ['credibility', 'emotional_appeal', 'logical_coherence']
        for m in mods:
            desc_data[m] = df[m]
        # Add control variables
        for c in CONTROLS:
            desc_data[c] = df[c]
        # Add source dummies if they exist
        if SOURCE_COL in df.columns:
            source_dummies = pd.get_dummies(df[SOURCE_COL], prefix="source", drop_first=True, dtype=int)
            desc_data = pd.concat([desc_data, source_dummies], axis=1)

    elif model_type == "controls":
        desc_data = pd.DataFrame()
        desc_data[TARGET] = df[TARGET]
        for c in CONTROLS:
            desc_data[c] = df[c]
        if SOURCE_COL in df.columns:
            source_dummies = pd.get_dummies(df[SOURCE_COL], prefix="source", drop_first=True, dtype=int)
            desc_data = pd.concat([desc_data, source_dummies], axis=1)
    elif model_type == "null":
        desc_data = pd.DataFrame({TARGET: df[TARGET]})
    else:
        raise ValueError("model_type must be 'full', 'controls', or 'null'")

    # Calculate descriptive statistics
    print("\n=== Descriptive Statistics ===")
    print("\nBasic Statistics:")
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Don't wrap wide tables
    print(desc_data.describe().round(3))

    # Calculate correlation matrix for numeric columns
    numeric_data = desc_data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 1:  # Only calculate correlations if we have multiple numeric columns
        print("\nCorrelation Matrix:")
        corr_matrix = numeric_data.corr().round(3)

        # Define the desired order of variables
        independent_var = 'x'  # The main independent variable
        moderators = ['credibility', 'emotional_appeal', 'logical_coherence']
        control_vars = [col for col in corr_matrix.columns
                        if col not in [TARGET, independent_var] + moderators]

        # Create the ordered list of columns
        ordered_cols = ([TARGET] +  # Target variable first
                        [independent_var] +  # Independent variable second
                        moderators +  # Moderator variables third
                        control_vars)  # Control variables last

        # Reorder the correlation matrix
        corr_matrix = corr_matrix[ordered_cols].reindex(ordered_cols)

        # Replace NaN values with empty strings for better readability
        corr_matrix = corr_matrix.fillna('')
        print(corr_matrix.to_string())

        # Calculate VIF for numeric columns (excluding the target variable)
        print("\nVariance Inflation Factors (VIF):")
        # Create a copy of the data without the target variable for VIF calculation
        vif_data = numeric_data.drop(columns=[TARGET]).copy()
        # Add constant term for VIF calculation
        vif_data['const'] = 1.0
        vif_data = calculate_vif(vif_data, verbose=True)

    # Print sample sizes and success rates
    print(f"\nSample Size: {len(desc_data)}")
    print(f"Number of Features: {len(desc_data.columns)}")

    # Print pitch success statistics
    success_rate = df[TARGET].mean()
    print(f"\nPitch Success Statistics:")
    print(f"Total Pitches: {len(df)}")
    print(f"Successful Pitches: {df[TARGET].sum()}")
    print(f"Failed Pitches: {len(df) - df[TARGET].sum()}")
    print(f"Success Rate: {success_rate:.1%}")


if __name__ == '__main__':
    # calculate_significant_results()
    print(calculate_sample(model_type="full").summary())
    # print(calculate_sample(include_independent=False).summary())
    # print_glm_stats(calculate_sample(model_type="full"), calculate_sample(model_type="controls"))
    # get_descriptive_statistics(model_type="full")