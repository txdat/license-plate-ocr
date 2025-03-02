import os
import cv2

data_dir = "../../data/LP"
label_dir = "../../data/LP_detect_labels1"

# label_path = os.listdir(label_dir)[0]

# img = cv2.imread(f"{data_dir}/{label_path.rsplit('.',1)[0]}")
# h, w, _ = img.shape

# with open(f"{label_dir}/{label_path}", mode="r") as f:
#     for line in f:
#         _, x, y, bw, bh = [float(x) for x in line.strip().split('\t')]
#         x = int(x * w)
#         y = int(y * h)
#         bw = int(bw * w)
#         bh = int(bh * h)
#         x0 = x - bw//2
#         y0 = y - bh//2
#         x1 = x0 + bw
#         y1 = y0 + bh
#         cv2.rectangle(img, (x0,y0), (x1,y1), (0,255,0), 3)

# img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
# cv2.imshow("test", img)
# cv2.waitKey(0)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

probs = []
for label_path in os.listdir(label_dir):
    with open(f"{label_dir}/{label_path}", mode="r") as f:
        for line in f:
            prob = float(line.strip().split("\t")[-1])
            probs.append(prob)

df = pd.DataFrame()
df["prob"] = np.array(probs).astype(np.float32)

sb.displot(df, x="prob", kind="kde")
plt.show()
