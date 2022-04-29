from dask_ml.model_selection import HyperbandSearchCV
from scikeras.wrappers import KerasClassifier
from scipy.stats import loguniform, uniform
from CIFAR10_CNN import cifar10_cnn

class cifar10_cnn_distributed(cifar10_cnn):
    
    def define_model(self, lr, momentum):
        niceties = dict(verbose=False)
        model = KerasClassifier(model=super().define_model(lr, momentum), **niceties)
        
        return model
    
    def fit_model(self, trainX, trainY, epochs=100, batch_size=64):
        params = {"lr": loguniform(1e-3, 1e-1), "momentum": uniform(0, 1)}
        search = HyperbandSearchCV(model, params, max_iter=27)
        search.fit(trainX, trainY, max_iter=epochs)
    
    def save_model_to_file(self):
        self.model.model.save(self.modelFilename)
