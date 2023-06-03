# Implementation-of-SMU-activation-in-tensorflow-2.0

```
class SMUActivation(tf.keras.layers.Layer):
    '''
    Implementation of SMU activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha: hyper parameter
    References:
        - See related paper:
        https://arxiv.org/abs/2111.04682
    '''
    def __init__(self, alpha=0.25, mu=1.0, trainable=True, **kwargs):
        super(SMUActivation, self).__init__(**kwargs)
        self.alpha = alpha
        self.mu = mu
        self.trainable = trainable

    def build(self, input_shape):
        self.mu = self.add_weight(
            name='mu',
            shape=(),
            initializer=tf.constant_initializer(self.mu),
            dtype=tf.float32,
            trainable=True
        )

        if self.trainable:
            self._trainable_weights.append(self.mu)

        super(SMUActivation, self).build(input_shape)

    def call(self, inputs):
        return ((1 + self.alpha) * inputs + (1 - self.alpha) * inputs * tf.math.erf(self.mu * (1 - self.alpha) * inputs)) / 2

    def get_config(self):
        config = { 
            'alpha': self.alpha,
            'mu': self.get_weights()[0] if self.trainable else self.mu,
            'trainable': self.trainable
          }
        base_config = super(SMUActivation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape 
 ```
       
