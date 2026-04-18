# Retail-Price-Optimization

============================================================
  MODULE 1 — PRICE ELASTICITY (Log-Log OLS)
============================================================

  Log-Log Regression:  log(qty) = α + ε·log(price)
  ─────────────────────────────────────────────────
  Intercept  (α)  : +2.7681   SE=0.2807  p=0.0000
  Elasticity (ε)  : -0.1313   SE=0.0623  p=0.0355
  R²              : 0.0065

  INTERPRETATION:
  → A 1% price increase → -0.13% change in demand
  → Demand is INELASTIC (|ε|<1): customers less price-sensitive

  PER-CATEGORY ELASTICITIES:
  Category                         Elasticity       R²     N
  ---------------------------------------------------------
  bed_bath_table                       +0.574    0.066    61
  computers_accessories                +0.398    0.009    69
  consoles_games                       -2.276    0.274    22
  cool_stuff                           +0.137    0.003    57
  furniture_decor                      +0.308    0.012    48
  garden_tools                         -0.799    0.086   160
  health_beauty                        -0.144    0.023   130
  perfumery                            -0.580    0.076    26
  watches_gifts                        -0.316    0.015   103


  ============================================================
  MODULE 2 — SIMPLE REVENUE-OPTIMAL PRICE
============================================================

  Revenue = Price × Quantity
  Using elasticity model: Q(p) = exp(α) × p^ε
  → Revenue(p) = p × exp(α) × p^ε = exp(α) × p^(1+ε)
  → dR/dp = 0  →  p* = 0 (if ε<-1, revenue maximized at 0)
  → Real world: Lerner condition  p* = MC / (1 + 1/ε)

  Category                         Elasticity   MC Proxy  Optimal Price   Avg Actual
  --------------------------------------------------------------------------------
  bed_bath_table                       +0.574      16.14          78.63†        78.63
  computers_accessories                +0.398      25.10         119.48†       119.48
  consoles_games                       -2.276      14.81          26.42        27.03
  cool_stuff                           +0.137      18.98         107.86†       107.86
  furniture_decor                      +0.308      16.94          60.15†        60.15
  garden_tools                         -0.799      28.46          80.09*        80.09
  health_beauty                        -0.144      18.61         132.31*       132.31
  perfumery                            -0.580      14.34          89.35*        89.35
  watches_gifts                        -0.316      16.49         164.88*       164.88

  * Inelastic: increase price (pricing power region)
  † Non-negative elasticity: investigate data quality / other drivers