from os import sys, path

import numpy as np

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import dataset_utils as du
import pytorch_kit.models as tm
import image_utils as iu

if __name__ == "__main__":
    np.random.seed(0)

    X, y = du.load_dataset("boat_images", as_image=False)
    
    # TRAIN NETWORK
    model = tm.AttentionModel(n_channels=3, n_outputs=1)
    model.fit(X, y, batch_size=23, epochs=150)

    show = lambda m, i: iu.show(m.get_heatmap(X)[i], X[i])
    show(model, 1)

    import pdb; pdb.set_trace()  # breakpoint c95ec4e5 //


