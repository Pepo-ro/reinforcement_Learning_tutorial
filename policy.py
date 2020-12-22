from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import linear

class TransformerPolicy(ActorCriticPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, feature_extraction="vision_transformer", reuse=False, **kwargs):
        super(TransformerPolicy, self).__init__(sess, ob_space, ac_space, n_env,
                                                n_steps, n_batch, reuse)
        
        self._kwargs_check(feature_extraction, kwargs)

        hidden_size = 32
        mlp_dim = 32
        num_layers = 4
        dff = 32#mlp_dim * num_layers 

        vison_transformer_params ={
            "classifier": "token",
            "hidden_size": hidden_size,
            "patches": [4, 4],
            "representation_size": None,
            "transformer": {
                "attention_dropout_rate": 0.0,
                "dropout_rate": 0.1,
                "mlp_dim": mlp_dim,
                "num_heads": 4,
                "num_layers": num_layers,
                "dff": dff
            }
        }

        with tf.variable_scope("model", reuse=reuse):
            pi_latent = vf_latent = vision_transform(self.processed_obs, vison_transformer_params, **kwargs)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()


    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})