\ Model Choquet
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0 y[0] + 0.0442996012090079 y[1] + 0.6687645417099455 y[2]
   + 0.2869358570810466 y[3]
Subject To
 budget: 40 x[0] + 50 x[1] + 60 x[2] + 50 x[3] <= 100
 y_1_0: - y[1] + 19 x[0] + 6 x[1] + 17 x[2] + 2 x[3] >= 0
 y_2_1: - y[2] + 2 x[0] + 11 x[1] + 4 x[2] + 18 x[3] >= 0
 y_3_0: - y[3] + 19 x[0] + 6 x[1] + 17 x[2] + 2 x[3] >= 0
 y_3_1: - y[3] + 2 x[0] + 11 x[1] + 4 x[2] + 18 x[3] >= 0
Bounds
Binaries
 x[0] x[1] x[2] x[3]
End
