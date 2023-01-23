\ Model Choquet
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.03864467522997 y[0] + 0.00878013194363005 y[1] + 0.0618779729030354 y[2]
   + 0.1885883093440415 y[3] + 0.1709554675983069 y[4]
   + 0.0378750496634688 y[5] + 0.00495461683633592 y[6]
   + 0.0959540910429402 y[7] + 0.0303791583573023 y[8]
   + 0.016199598471902 y[9] + 0.0266022536746559 y[10]
   + 0.1235893635247311 y[11] + 0.0175591989038171 y[12]
   + 0.0140911404884414 y[13] + 0.1432029286265807 y[14]
   + 0.0207460433908408 y[15]
Subject To
 budget: 40 x[0] + 50 x[1] + 60 x[2] + 50 x[3] <= 100
 aptitude_0: - y[0] >= 0
 aptitude_0: - y[0] >= 0
 aptitude_1: 19 x[0] - y[1] >= 0
 aptitude_1: 2 x[0] - y[1] >= 0
 aptitude_2: 6 x[1] - y[2] >= 0
 aptitude_2: 11 x[1] - y[2] >= 0
 aptitude_3: 17 x[2] - y[3] >= 0
 aptitude_3: 4 x[2] - y[3] >= 0
 aptitude_4: 2 x[3] - y[4] >= 0
 aptitude_4: 18 x[3] - y[4] >= 0
 aptitude_5: 19 x[0] + 6 x[1] - y[5] >= 0
 aptitude_5: 2 x[0] + 11 x[1] - y[5] >= 0
 aptitude_6: 19 x[0] + 17 x[2] - y[6] >= 0
 aptitude_6: 2 x[0] + 4 x[2] - y[6] >= 0
 aptitude_7: 19 x[0] + 2 x[3] - y[7] >= 0
 aptitude_7: 2 x[0] + 18 x[3] - y[7] >= 0
 aptitude_8: 6 x[1] + 17 x[2] - y[8] >= 0
 aptitude_8: 11 x[1] + 4 x[2] - y[8] >= 0
 aptitude_9: 6 x[1] + 2 x[3] - y[9] >= 0
 aptitude_9: 11 x[1] + 18 x[3] - y[9] >= 0
 aptitude_10: 17 x[2] + 2 x[3] - y[10] >= 0
 aptitude_10: 4 x[2] + 18 x[3] - y[10] >= 0
 aptitude_11: 19 x[0] + 6 x[1] + 17 x[2] - y[11] >= 0
 aptitude_11: 2 x[0] + 11 x[1] + 4 x[2] - y[11] >= 0
 aptitude_12: 19 x[0] + 6 x[1] + 2 x[3] - y[12] >= 0
 aptitude_12: 2 x[0] + 11 x[1] + 18 x[3] - y[12] >= 0
 aptitude_13: 19 x[0] + 17 x[2] + 2 x[3] - y[13] >= 0
 aptitude_13: 2 x[0] + 4 x[2] + 18 x[3] - y[13] >= 0
 aptitude_14: 6 x[1] + 17 x[2] + 2 x[3] - y[14] >= 0
 aptitude_14: 11 x[1] + 4 x[2] + 18 x[3] - y[14] >= 0
 aptitude_15: 19 x[0] + 6 x[1] + 17 x[2] + 2 x[3] - y[15] >= 0
 aptitude_15: 2 x[0] + 11 x[1] + 4 x[2] + 18 x[3] - y[15] >= 0
Bounds
Binaries
 x[0] x[1] x[2] x[3]
End
