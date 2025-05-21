import tensorflow as tf
from tensorflow.keras import layers

# %%   
"""
Code and initialisation for AE NN

"""

class MLP(layers.Layer):
    def __init__(self, units, activation):
        super().__init__()
        
        self.ls = []
        
        for (u, a, ) in zip(units, activation):
                
            self.ls += [layers.Dense(u, a)]      
       
    def __call__(self, spectra_in):     
         
        x = spectra_in 
        for l in self.ls:
            x = l(x)
        return x
    
    
def main(**kwargs):
    spectra_in = tf.keras.Input(shape=[1000])    
    spectra_out = MLP(**kwargs)(spectra_in)    
    model = tf.keras.Model(inputs = [spectra_in], outputs=[spectra_out])
    model.compile('adam', 'mse')
    return model






