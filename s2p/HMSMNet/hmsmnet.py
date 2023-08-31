import os
import numpy as np
from tensorflow import keras
from feature import FeatureExtraction
from cost import CostConcatenation
from aggregation import Hourglass, FeatureFusion
from computation import Estimation
from refinement import Refinement
from data_reader import read_left, read_right
import rasterio

import argparse


# mindisp = -128
# maxdisp = 64

class HMSMNet:
    def __init__(self, height=1024, width=1024, channel=3, min_disp=-128, max_disp=64):
        self.height = height
        self.width = width
        self.channel = channel
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.model = None

    def build_model(self):
        left_image = keras.Input(shape=(self.height, self.width, self.channel))
        right_image = keras.Input(shape=(self.height, self.width, self.channel))
        gx = keras.Input(shape=(self.height, self.width, self.channel))
        gy = keras.Input(shape=(self.height, self.width, self.channel))

        feature_extraction = FeatureExtraction(filters=16)
        [l0, l1, l2] = feature_extraction(left_image)
        [r0, r1, r2] = feature_extraction(right_image)

        cost0 = CostConcatenation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        cost1 = CostConcatenation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        cost2 = CostConcatenation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16)
        cost_volume0 = cost0([l0, r0])
        cost_volume1 = cost1([l1, r1])
        cost_volume2 = cost2([l2, r2])

        hourglass0 = Hourglass(filters=16)
        hourglass1 = Hourglass(filters=16)
        hourglass2 = Hourglass(filters=16)
        agg_cost0 = hourglass0(cost_volume0)
        agg_cost1 = hourglass1(cost_volume1)
        agg_cost2 = hourglass2(cost_volume2)

        estimator2 = Estimation(min_disp=self.min_disp // 16, max_disp=self.max_disp // 16)
        disparity2 = estimator2(agg_cost2)

        fusion1 = FeatureFusion(units=16)
        fusion_cost1 = fusion1([agg_cost2, agg_cost1])
        hourglass3 = Hourglass(filters=16)
        agg_fusion_cost1 = hourglass3(fusion_cost1)

        estimator1 = Estimation(min_disp=self.min_disp // 8, max_disp=self.max_disp // 8)
        disparity1 = estimator1(agg_fusion_cost1)

        fusion2 = FeatureFusion(units=16)
        fusion_cost2 = fusion2([agg_fusion_cost1, agg_cost0])
        hourglass4 = Hourglass(filters=16)
        agg_fusion_cost2 = hourglass4(fusion_cost2)

        estimator0 = Estimation(min_disp=self.min_disp // 4, max_disp=self.max_disp // 4)
        disparity0 = estimator0(agg_fusion_cost2)

        # refinement
        refiner = Refinement(filters=32)
        final_disp = refiner([disparity0, left_image, gx, gy])

        self.model = keras.Model(inputs=[left_image, right_image, gx, gy],
                                 outputs=[disparity2, disparity1, disparity0, final_disp])
        self.model.summary()

    def predict(self, left_dir, right_dir, output_dir, weights):
        self.model.load_weights(weights)
        left_image, gx, gy = read_left(left_dir)
        right_image = read_right(right_dir)
        left_image = np.expand_dims(left_image, 0)
        gx = np.expand_dims(gx, 0)
        gy = np.expand_dims(gy, 0)
        right_image = np.expand_dims(right_image, 0)

        disparity = self.model.predict([left_image, right_image, gx, gy])
        disparity = disparity[-1][0, :, :, 0]

        with rasterio.open(output_dir, 'w', 'GTiff', width=disparity.shape[1],
                           height=disparity.shape[0], count=1, dtype=disparity.dtype
                           ) as dst: dst.write(disparity, 1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("S2P-HMSMNet")
    # parser.add_argument("--mindisp", required=True, type=int)
    # parser.add_argument("--maxdisp", required=True, type=int)
    # parser.add_argument("--left_dir", required=True)
    # parser.add_argument("--right_dir", required=True)
    # parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default="/home/yx/MyCode/DL-3DConstruction/S2P_DL/dl_models/HMSMNet/US3D.h5")
    args = parser.parse_args()

    # img = read_right(args.right_dir)
    # height = img.shape[0]
    # width = img.shape[1]
    # if img.ndim == 2:
    #     channel = 1
    # else:
    #     channel = 3

    # net = HMSMNet(height, width, channel, args.mindisp, args.maxdisp)
    # net.build_model()
    # net.predict(args.left_dir, args.right_dir, args.output_dir, args.weights)


    net = HMSMNet(1024, 1024, 3, 0, 96)
    net.build_model()
    net.predict('left.tif', 'right.tif', 'disp.tif', args.weights)