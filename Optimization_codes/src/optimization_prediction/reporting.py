from __future__ import annotations

from pathlib import Path

import pandas as pd


def _markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        frame = frame.head(max_rows)
    if frame.empty:
        return "_No rows._"
    view = frame.copy()
    for column in view.columns:
        if pd.api.types.is_datetime64_any_dtype(view[column]):
            view[column] = view[column].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.4f}")
    headers = [str(column) for column in view.columns]
    separator = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(separator) + " |"]
    for row in view.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _comparison_delta_table(previous_metrics: pd.DataFrame, current_metrics: pd.DataFrame) -> pd.DataFrame:
    if previous_metrics.empty:
        return pd.DataFrame()
    common_columns = [
        column
        for column in [
            "mae",
            "rmse",
            "wape",
            "bias",
            "underprediction_units",
            "overprediction_units",
            "purchase_cost_proxy",
            "holding_cost_proxy",
        ]
        if column in previous_metrics.columns and column in current_metrics.columns
    ]
    previous = previous_metrics.iloc[0]
    current = current_metrics.iloc[0]
    rows = []
    for column in common_columns:
        rows.append(
            {
                "metric": column,
                "previous_model": float(previous[column]),
                "upgraded_model": float(current[column]),
                "delta_new_minus_old": float(current[column] - previous[column]),
            }
        )
    return pd.DataFrame(rows)


def _image_markdown(path: str, caption: str) -> str:
    clean_path = path.replace("\\", "/")
    return f"![{caption}]({clean_path})"


def _select_columns(frame: pd.DataFrame, mapping: list[tuple[str, str]], max_rows: int | None = None) -> pd.DataFrame:
    selected = frame[[source for source, _ in mapping]].copy()
    selected.columns = [target for _, target in mapping]
    if max_rows is not None:
        selected = selected.head(max_rows)
    return selected


def render_markdown_report(context: dict) -> str:
    train_dates = f"{context['train_start'].date()} to {context['train_end'].date()}"
    test_dates = f"{context['test_start'].date()} to {context['test_end'].date()}"
    previous_vs_new = _comparison_delta_table(context["previous_metrics"], context["best_model_summary"])
    best_summary = context["best_model_summary"].iloc[0]
    optimization_summary = context["optimization_summary_table"]
    optimization_gap = optimization_summary[optimization_summary["scenario"] == "Realized gap vs oracle"].iloc[0]
    direct_models = context["model_comparison"][context["model_comparison"]["approach"] == "direct_warehouse"].copy()
    allocation_models = context["model_comparison"][context["model_comparison"]["approach"] == "cluster_to_warehouse"].copy()
    direct_best = direct_models.sort_values("test_total_proxy_cost").iloc[0]
    allocation_best = allocation_models.sort_values("test_total_proxy_cost").iloc[0]
    direct_lag = context["model_comparison"][context["model_comparison"]["model_id"] == "direct_warehouse__lag_1"].iloc[0]
    direct_ma3 = context["model_comparison"][context["model_comparison"]["model_id"] == "direct_warehouse__moving_average_3"].iloc[0]
    direct_ma7 = context["model_comparison"][context["model_comparison"]["model_id"] == "direct_warehouse__moving_average_7"].iloc[0]
    direct_rf = context["model_comparison"][context["model_comparison"]["model_id"] == "direct_warehouse__random_forest"].iloc[0]
    direct_gb = context["model_comparison"][context["model_comparison"]["model_id"] == "direct_warehouse__gradient_boosting"].iloc[0]
    direct_lr = context["model_comparison"][context["model_comparison"]["model_id"] == "direct_warehouse__linear_regression"].iloc[0]
    direct_ridge = context["model_comparison"][context["model_comparison"]["model_id"] == "direct_warehouse__ridge_alpha_1.0"].iloc[0]
    direct_lasso = context["model_comparison"][context["model_comparison"]["model_id"] == "direct_warehouse__lasso_alpha_0.0100"].iloc[0]
    forecast_comparison = _select_columns(
        context["model_comparison"],
        [
            ("selection_rank", "rank"),
            ("approach", "approach"),
            ("model_name", "model"),
            ("family", "family"),
            ("validation_wape", "val_wape"),
            ("validation_bias", "val_bias"),
            ("validation_total_proxy_cost", "val_proxy_cost"),
            ("test_wape", "test_wape"),
            ("test_rmse", "test_rmse"),
            ("test_bias", "test_bias"),
            ("test_total_proxy_cost", "test_proxy_cost"),
        ],
    )
    cluster_snapshot = _select_columns(
        context["best_cluster_metrics"],
        [
            ("sku_cluster_ID", "cluster"),
            ("actual_units", "actual_units"),
            ("predicted_units", "predicted_units"),
            ("wape", "wape"),
            ("bias", "bias"),
            ("underprediction_units", "underprediction_units"),
            ("overprediction_units", "overprediction_units"),
            ("purchase_cost_proxy", "purchase_cost_proxy"),
        ],
        max_rows=12,
    )
    warehouse_snapshot = _select_columns(
        context["best_warehouse_metrics"],
        [
            ("warehouse", "warehouse"),
            ("actual_units", "actual_units"),
            ("predicted_units", "predicted_units"),
            ("wape", "wape"),
            ("bias", "bias"),
            ("underprediction_units", "underprediction_units"),
            ("overprediction_units", "overprediction_units"),
            ("purchase_cost_proxy", "purchase_cost_proxy"),
        ],
        max_rows=12,
    )
    learned_importance = _select_columns(
        context["learned_feature_importance"],
        [
            ("feature", "feature"),
            ("importance", "importance"),
            ("importance_type", "importance_type"),
        ],
        max_rows=10,
    )
    daily_comparison = _select_columns(
        context["daily_optimization_comparison"],
        [
            ("date", "date"),
            ("predicted_policy_cost", "pred_cost"),
            ("predicted_policy_shortage_units", "pred_shortage"),
            ("predicted_policy_overflow_units", "pred_overflow"),
            ("predicted_policy_service_level", "pred_service"),
            ("oracle_cost", "oracle_cost"),
            ("oracle_shortage_units", "oracle_shortage"),
            ("oracle_overflow_units", "oracle_overflow"),
            ("oracle_service_level", "oracle_service"),
            ("cost_gap_vs_oracle", "cost_gap"),
        ],
    )
    price_summary = context.get("price_summary", pd.DataFrame())
    capacity_summary = context.get("capacity_summary", pd.DataFrame())
    parameter_update_table = context.get("parameter_update_table", pd.DataFrame())
    capacity_sample = pd.DataFrame()
    if not capacity_summary.empty:
        capacity_sample = (
            capacity_summary[["warehouse", "legacy_capacity", "corrected_capacity", "C_j", "capacity_source"]]
            .sort_values("C_j", ascending=False)
            .head(12)
            .reset_index(drop=True)
        )

    lines = [
        "# Optimization Model Report",
        "",
        "## Executive Summary",
        f"- Built a complete forecast-to-optimization system that predicts warehouse-cluster demand, reconstructs the notebook parameters, solves a downstream inventory-transfer LP, and evaluates the resulting operating decisions against realized demand.",
        f"- The selected forecast model was **`{context['best_model_label']}`**. On the March 25-31, 2018 test week it achieved **WAPE = {best_summary['wape']:.4f}**, **RMSE = {best_summary['rmse']:.4f}**, and **bias = {best_summary['bias']:.4f}**.",
        f"- When that forecast was used inside the optimizer, the realized weekly cost gap versus the oracle policy was **{optimization_gap['total_cost']:.2f}**, with a service-level gap of **{optimization_gap['service_level']:.4f}**.",
        "- The main remaining weakness is the missing original SKU-to-cluster bridge, which still limits how precisely prices, inventory, and behavioral features can be attached to the provided cluster IDs.",
        "",
        "## Objective",
        "The capstone objective is not simply to predict demand. The practical goal is to support inventory and fulfillment decisions across a multi-warehouse network. This system therefore handles both parts of the workflow: it forecasts daily demand by warehouse and cluster, then sends those forecasts into an optimization model that chooses procurement, transfers, ending inventory, and shortage exposure.",
        "",
        "## Alignment with Meeting Notes",
        "The meeting notes and capstone documents shaped the modeling choices below. The final system follows those expectations directly.",
        "",
        _markdown_table(pd.DataFrame(context["meeting_notes_alignment"])),
        "",
        "## Data and Files Used",
        _markdown_table(pd.DataFrame(context["files_used"])),
        "",
        "## Parameter Construction",
        "The price and capacity parameters were updated in this revision because the team confirmed that the original notebook logic was not fully reliable for either one. Those corrections propagate into every weighted proxy cost and into the optimizer's capacity constraints.",
        "",
        "### Corrected Price Construction",
        "The notebook comments suggested dividing `original_unit_price` by quantity, but the executable notebook line did not actually do that. The corrected implementation treats `original_unit_price` as the unit price, computes an average SKU price over the training orders, and then aggregates those SKU prices to cluster-level `p_s` using the available SKU-to-cluster bridge.",
        "This correction matters because `p_s` enters every weighted shortage, holding, transfer, and forecast proxy cost term. A bad unit-price definition would distort the relative importance of clusters even if the demand forecasts themselves were unchanged.",
        _markdown_table(price_summary),
        "",
        "### Corrected Warehouse Capacity Construction",
        "The updated `C_j` values are now constructed from `JD_daily_sku_dc_summary.xlsx` by summing `inventory` across all SKUs for each warehouse-day and then taking the maximum observed warehouse-day total for each warehouse. When that corrected series is missing or entirely nonpositive for a warehouse, the pipeline falls back to the legacy capacity file instead of forcing an artificial zero-capacity node into the LP.",
        "This corrected logic is more defensible because it uses all available SKU inventory activity rather than relying only on a static file that the team flagged as likely based on 1P SKU. It also avoids treating missing inventory history as literal zero capacity.",
        _markdown_table(capacity_sample),
        "",
        "### Parameter Corrections in This Revision",
        "The table below compares the pre-correction parameterized outputs against the regenerated version after the price and capacity updates.",
        _markdown_table(parameter_update_table),
        "",
        "## Train/Test Split",
        f"- Training dates: **{train_dates}**",
        f"- Testing dates: **{test_dates}**",
        "- The split is date-based and matches the local JD order horizon and the provided train/test demand files.",
        "- The test file omitted cluster `2`; the pipeline restored those warehouse-cluster-date rows explicitly as zeros after validating warehouse-date totals against raw orders.",
        "",
        "## Forecasting System",
        "The forecasting stage compares the two modeling levels discussed in the project meetings and then tests several model families inside each level.",
        "- **Direct warehouse-cluster prediction** forecasts each `(cluster, warehouse)` daily series directly.",
        "- **Cluster total -> warehouse allocation** first forecasts cluster totals and then allocates them using empirical warehouse shares estimated from the real training demand panel.",
        "- Feature engineering includes lagged demand, rolling demand averages, day-of-week, warehouse and cluster identifiers, lagged promotion proxies from orders, and lagged click signals.",
        "- Forecast evaluation is rolling one-day-ahead within the validation and test windows, which matches the overnight planning cadence described in the meeting notes.",
        "",
        "### Direct Warehouse-Cluster Models",
        "This approach predicts demand exactly where the optimizer needs it: at the warehouse-cluster level. Intuitively, it asks the model to learn the local demand pattern for each warehouse rather than first averaging that pattern away and then reconstructing it later.",
        "It was considered because the downstream optimization acts at the warehouse level, so preserving local demand differences is valuable if the data supports it.",
        f"In this project, the direct approach was consistently stronger. Its best test-week result was **WAPE = {direct_best['test_wape']:.4f}**, while the best cluster-allocation alternative finished at **WAPE = {allocation_best['test_wape']:.4f}**. The top three ranked models were all direct warehouse models, which suggests that the warehouse-specific signal was worth keeping instead of re-allocating after the forecast step.",
        "Its main weakness is dimensionality: once demand is modeled directly by warehouse and cluster, the number of series grows quickly and the history per series becomes very short. That makes complex models harder to stabilize.",
        "",
        "### Cluster to Warehouse Allocation Approach",
        "This approach first forecasts total demand by cluster and only afterward splits that total across warehouses using learned warehouse shares. The intuition is that cluster demand may be easier to forecast in aggregate because it is less sparse and less noisy.",
        "It was considered because it reduces the forecasting problem to a smaller number of series and can be attractive when warehouse shares are stable over time.",
        f"In this dataset, that extra allocation step usually hurt more than it helped. The best cluster-allocation model still trailed the selected direct model on both error and proxy cost, with **test WAPE = {allocation_best['test_wape']:.4f}** versus **{direct_best['test_wape']:.4f}**. The likely reason is that warehouse shares were not stable enough over such a short horizon, so the second-stage allocation introduced avoidable error.",
        "Its strength is parsimony. Its weakness here is that it smooths away the local warehouse signal that the optimizer ultimately needs.",
        "",
        "### Baseline Models",
        "The baseline models are intentionally simple: lag-1, moving averages, and the 50/50 blend. Intuitively, these methods assume that the very recent past contains most of the useful information for the next day.",
        "They were included for two reasons. First, the meeting notes explicitly called for clear benchmark models. Second, with only a short March 2018 horizon, simple recency rules are a realistic competitor to heavier machine-learning models.",
        f"The results support that choice. Pure lag-1 was responsive but too noisy: the direct lag model reached **test WAPE = {direct_lag['test_wape']:.4f}**. The 7-day moving average was smoother but lagged behind changes in level, which showed up as more underprediction and a much more negative bias (**test bias = {direct_ma7['test_bias']:.4f}**). The 3-day moving average was more competitive (**test WAPE = {direct_ma3['test_wape']:.4f}**), but it still reacted more slowly than the final blend.",
        "In this context, the baselines worked because demand was short-horizon, operational, and strongly driven by the most recent observations. Their main weakness is that they cannot learn richer nonlinear interactions or broader calendar structure if those signals exist.",
        "",
        "### Linear Models",
        "The linear models try to explain next-day demand as a weighted combination of lagged demand, rolling features, identifiers, and other structured signals. Ridge and lasso add regularization so the model does not rely too heavily on unstable coefficients.",
        "They were considered because they are interpretable, fast, and often strong tabular baselines when there is enough history to estimate the coefficients reliably.",
        f"Here, they were not well calibrated. The direct linear regression model finished with **test WAPE = {direct_lr['test_wape']:.4f}** and a strongly negative **test bias = {direct_lr['test_bias']:.4f}**. Ridge and lasso were even less stable in this setup; for example, direct ridge had **validation bias = {direct_ridge['validation_bias']:.4f}**, while direct lasso still carried **test WAPE = {direct_lasso['test_wape']:.4f}**. That pattern suggests coefficient estimation was too fragile for the short training window and the wide, mostly sparse design matrix created by warehouse and cluster identifiers.",
        "Their strength is transparency. Their weakness on this dataset is that they need more stable signal and more history than were available here.",
        "",
        "### Tree-Based Models",
        "The tree models try to learn nonlinear thresholds and interactions automatically. In intuitive terms, they can capture patterns like 'this warehouse behaves differently when recent demand or promotions cross a certain level' without requiring that structure to be specified manually.",
        "They were considered because this is a tabular forecasting problem with mixed numeric and categorical features, which is usually a good setting for random forests and boosting.",
        f"They were competitive, but not quite good enough on the metrics that mattered most. The direct random forest achieved a lower **test RMSE = {direct_rf['test_rmse']:.4f}** than the selected blend, which means it sometimes tracked the shape of demand reasonably well. But it also carried worse calibration and higher operational proxy cost, with **test WAPE = {direct_rf['test_wape']:.4f}** and **test bias = {direct_rf['test_bias']:.4f}**. Gradient boosting was less stable still, ending at **test WAPE = {direct_gb['test_wape']:.4f}**.",
        "In this context, the tree models likely picked up some real short-run structure, but the data window was too small to support fully reliable calibration across all warehouse-cluster series.",
        "",
        "### Blended Models",
        "The selected blend combines two useful but imperfect ideas: yesterday matters, and a short moving average helps reduce noise. The model assigns equal weight to lag-1 and the 3-day average.",
        "It was considered because it offers a practical compromise between reactiveness and smoothing. If the lag model jumps too much and the moving average reacts too slowly, a blend can often land in the middle.",
        f"That is exactly what happened here. The blend outperformed the direct lag model (**{best_summary['wape']:.4f}** vs **{direct_lag['test_wape']:.4f}** test WAPE) and also beat the direct 3-day and 7-day moving averages on overall proxy cost. It kept enough recency to respond to week-to-week changes while removing some of the one-day volatility that hurt the pure lag forecast.",
        "Its main strength is robustness. Its weakness is that it remains a short-memory rule and cannot exploit deeper exogenous structure unless the feature set or data history expands.",
        "",
        "### Why the Simple Model Won",
        f"The short answer is that this dataset rewarded stability more than flexibility. The selected `blend_50_50` model finished with **test WAPE = {best_summary['wape']:.4f}** and **test bias = {best_summary['bias']:.4f}**, which was a better operational balance than the more complex alternatives.",
        "There are three practical reasons for that result. First, the horizon is short: with only March 1-24 available for training, the richer models did not have enough repeated history to estimate many warehouse-cluster-specific effects cleanly. Second, the problem is noisy: day-to-day demand varies enough that highly flexible models can chase local fluctuations that do not repeat in the test week. Third, the optimization layer cares about calibration as much as point accuracy. A model can improve RMSE on some rows and still create worse purchasing decisions if it shifts the total forecast too low or too high.",
        f"The results match that story. Random forest improved RMSE but still underperformed on WAPE and bias, while the 7-day moving average smoothed too aggressively and drifted into stronger underprediction. The 50/50 blend worked because it sat at the right point on the noise-versus-signal trade-off: more stable than lag-1, more responsive than a pure moving average, and less fragile than the learned models on this amount of data.",
        "",
        "### Forecast Comparison",
        "Models were first screened on validation calibration (`|bias| <= 0.15` and `WAPE <= 0.45`) and then ranked by validation optimization proxy cost. The full comparison table is also saved in `results/optimization_prediction/tables/model_comparison.csv`.",
        "",
        _markdown_table(forecast_comparison),
        "",
        "### Forecast Figures",
        _image_markdown(context["figure_paths"]["model_comparison_wape"], "Model comparison by test WAPE"),
        "",
        _image_markdown(context["figure_paths"]["best_model_daily_totals"], "Best model daily totals"),
        "",
        _image_markdown(context["figure_paths"]["best_model_daily_error"], "Best model daily error"),
        "",
        _image_markdown(context["figure_paths"]["best_model_residual_histogram"], "Best model residual distribution"),
        "",
        "### Feature Importance",
        "The selected model is a rule-based blend, so its importance is simply the forecast rule itself. For context, the report also shows the strongest learned benchmark.",
        "",
        "Selected model driver summary:",
        _markdown_table(context["selected_feature_importance"]),
        "",
        _image_markdown(context["figure_paths"]["selected_model_feature_importance"], "Selected model driver chart"),
        "",
        f"Best learned benchmark reviewed for interpretability: **`{context['best_learned_model_label']}`**",
        _markdown_table(learned_importance),
        "",
        _image_markdown(context["figure_paths"]["best_learned_model_feature_importance"], "Best learned model driver chart"),
        "",
        "### Forecast Error Analysis",
        f"- The selected forecast model produced **underprediction = {best_summary['underprediction_units']:.0f} units** and **overprediction = {best_summary['overprediction_units']:.0f} units** on the test week.",
        f"- The selected model's proxy costs were **purchase = {best_summary['purchase_cost_proxy']:.2f}** and **holding = {best_summary['holding_cost_proxy']:.2f}**.",
        "- The top-volume cluster and warehouse breakdowns are shown below; full detail remains available in the saved CSV outputs.",
        "",
        _markdown_table(cluster_snapshot),
        "",
        _markdown_table(warehouse_snapshot),
        "",
        "## Optimization Model",
        "After selecting the forecast model, the system solves a downstream linear program for each day of the test week using a receding-horizon state update. This keeps the optimization operationally realistic while leaving the route-level network formulation tractable on the available hardware.",
        "",
        "### Decision Variables",
        "- `q_{s,j,d}`: procurement quantity of cluster `s` into warehouse `j` on day `d`.",
        "- `x_{s,n,m,d}`: transfer quantity of cluster `s` from warehouse `n` to warehouse `m` on day `d`.",
        "- `I_{s,j,d}`: ending inventory of cluster `s` at warehouse `j` after day `d`.",
        "- `z_{s,j,d}`: shortage of cluster `s` at warehouse `j` on day `d`.",
        "",
        "### Objective Function",
        "For each day, the LP minimizes:",
        "- `sum_{s,j} p_s * u * z_{s,j,d}` shortage penalty",
        "- `sum_{s,j} p_s * h * I_{s,j,d}` holding cost",
        "- `sum_{s,n,m} p_s * k * t_{n,m} * x_{s,n,m,d}` transfer and delivery-time penalty",
        "",
        "### Constraints",
        "- Inventory balance: `I_{s,j,d} = I0_{s,j,d} + q_{s,j,d} + inflow_{s,j,d} - outflow_{s,j,d} - D_{s,j,d} + z_{s,j,d}`",
        "- Capacity: `sum_s I_{s,j,d} <= C_j`",
        "- Procurement eligibility: `q_{s,j,d} <= M_{s,j,d} W_j`",
        "- Route feasibility: `x_{s,n,m,d} <= M_{s,n,m,d} r_{n,m}`",
        "- Non-negativity: all decision variables are constrained to be nonnegative",
        "- Initial condition: the first-day inventory state is `I0_{s,j}` and later days inherit the previous realized ending inventory",
        "",
        "## Parameter Usage Audit",
        _markdown_table(context["parameter_usage_audit"]),
        "",
        "### Parameter Diagnostics",
        _markdown_table(pd.DataFrame(context["parameter_diagnostics"])),
        "",
        "## Decision Results",
        "The optimizer was run under two demand-information regimes:",
        "- **Predicted policy:** uses the selected forecast to choose procurement and transfer decisions, then those decisions are evaluated under actual realized demand.",
        "- **Oracle policy:** uses actual demand as a perfect-information benchmark.",
        "- Under the corrected and generally tighter capacity layer, the realized rollout can end a day above on-site capacity when actual demand comes in below the planning forecast. The implementation now records those overflow units explicitly and trims them from the next-day usable state starting with the lowest-value clusters, which keeps the receding-horizon LP feasible while making the capacity impact visible in the outputs.",
        "",
        _markdown_table(context["optimization_summary_table"]),
        "",
        "Daily comparison:",
        _markdown_table(daily_comparison),
        "",
        _image_markdown(context["figure_paths"]["optimization_cost_comparison"], "Optimization weekly cost comparison"),
        "",
        _image_markdown(context["figure_paths"]["optimization_daily_service_level"], "Optimization daily service level"),
        "",
        _image_markdown(context["figure_paths"]["optimization_daily_shortage"], "Optimization daily shortage"),
        "",
        "## What Changed from Previous Version",
        "- Replaced the previous synthetic demand construction with the provided cluster-by-warehouse demand files.",
        "- Removed the old KMeans demand logic from the forecasting target.",
        "- Corrected the cluster price construction so `p_s` now comes from SKU unit prices instead of the previous revenue-regression workaround.",
        "- Corrected the warehouse capacity construction so `C_j` now comes from the all-SKU daily inventory summary, with the old capacity file used only as fallback.",
        "- Added a real optimization layer so the pipeline now evaluates decisions, not only forecast error.",
        "- Enforced explicit mathematical use of `D`, `I0`, `W`, `C`, `r`, `t`, `p`, `u`, `h`, and `k` in the solver.",
        "",
        "### Previous vs Upgraded Best Model",
        _markdown_table(previous_vs_new),
        "",
        "## Key Insights",
        f"- The selected forecast model is simple, but on this short March 2018 horizon it generalized better than the more parameter-heavy learned alternatives.",
        f"- The optimization layer makes the forecast consequences concrete: the predicted policy still incurs a **weekly realized cost gap of {optimization_gap['total_cost']:.2f}** versus oracle information.",
        f"- The main operational risk remains shortage exposure. Even after optimization, the forecast-driven policy leaves more shortage than the oracle policy and therefore lower service on the hardest days.",
        "",
        "## Issues Found",
        _markdown_table(pd.DataFrame(context["issues_found"])),
        "",
        "## Remaining Limitations",
        "- The original SKU-to-cluster file is still missing, so price and inventory parameters remain partially inferred at aggregate level.",
        "- The optimization currently assumes same-day procurement and same-day transfer execution with time entering through cost, not through explicit shipment lead-time carryover.",
        "- Only one month of data is available, which limits the stability of richer forecasting models and any broader seasonality analysis.",
        "- The selected forecast model is a transparent rule-based blend; it works well here, but richer exogenous planning inputs are still missing.",
        "",
        "## Next Steps",
        "- Add multi-day lead-time inventory flow if JD can provide shipment timing needed for explicit carryover constraints.",
        "- Recover a defensible SKU-to-cluster bridge so prices, promotions, clicks, and inventory can be attached to the provided clusters directly.",
        "- Extend model selection to rolling-origin validation over more history once additional months become available.",
        "- Evaluate decision quality under alternative service targets or explicit stockout penalties agreed with the project sponsor.",
        "",
        "## Final Verdict",
        "This is no longer just a forecasting exercise. The project now behaves like a real planning prototype: it predicts demand at the warehouse-cluster level, turns those predictions into procurement and transfer decisions, and shows what forecast error costs once the network has to operate with corrected price and capacity inputs. The current version is credible as a capstone deliverable and useful for discussion with a professor or sponsor. The next improvement is also clear: better data fidelity, especially a raw SKU-to-cluster bridge and a cleaner inventory state, would likely do more for decision quality than adding yet another complex forecasting model.",
    ]
    return "\n".join(lines) + "\n"


def write_report_files(markdown_text: str, markdown_path: Path, html_path: Path | None = None) -> None:
    markdown_path.write_text(markdown_text, encoding="utf-8")
    if html_path is not None:
        html_path.write_text("<html><body><pre>See optimization_model_report.md</pre></body></html>", encoding="utf-8")
