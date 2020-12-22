import numpy as np
import tensorflow as tf
from stable_baselines.a2c.utils import linear,conv_to_fc


def vision_transformer(input_tensor, params, **kwargs):
    patch_emb = tf.layers.conv2d(
        input_tensor,
        params["hidden_size"],
        params["patches"],
        strides=params["patches"],
        padding='VALID',
        name='patch_embedding')

    transformer_params = params["transformer"]

    n, h, w, c = patch_emb.shape

    patch_emb =  tf.reshape(patch_emb, [-1, h * w, c])
  
    encoder = Encoder(
                        num_layers=transformer_params["num_layers"], 
                        mlp_dim=transformer_params["mlp_dim"], 
                        num_heads=transformer_params["num_heads"], 
                        dff=transformer_params["dff"],
                        dropout_rate=transformer_params["dropout_rate"],
                        attention_dropout_rate=transformer_params["attention_dropout_rate"]
                     )

    output = encoder(patch_emb, False)
    output = conv_to_fc(output)
    output = linear(output, 'full_conect',  n_hidden=params["hidden_size"], init_scale=np.sqrt(2))
    activ = tf.nn.relu

    return activ(output)

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, mlp_dim, num_heads, dff, dropout_rate, attention_dropout_rate, maximum_position_encoding=10000):
    super(Encoder, self).__init__()

    self.mlp_dim = mlp_dim
    self.num_layers = num_layers

    self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                            self.mlp_dim)
    self.enc_layers = [EncoderLayer(mlp_dim, num_heads, dff, attention_dropout_rate) 
                       for _ in range(num_layers)]
    #self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x, training, mask=None):

    seq_len = tf.shape(x)[1]

    pos_emb = self.pos_encoding[:, :seq_len, :]
    x += pos_emb

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x

def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, mlp_dim):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(mlp_dim)[np.newaxis, :],
                          mlp_dim)

  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


def point_wise_feed_forward_network(mlp_dim, out_dim=None):
  "point_wise_feed_forward_network"

  actual_out_dim = mlp_dim if out_dim is None else out_dim

  return tf.keras.Sequential([
      tf.keras.layers.Dense(mlp_dim, activation=gelu),
      tf.keras.layers.Dense(actual_out_dim)
  ])

def scaled_dot_product_attention(q, k, v, mask):
  matmul_qk = tf.matmul(q, k, transpose_b=True)


  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
  output = tf.matmul(attention_weights, v)

  return output, attention_weights

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, mlp_dim, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(mlp_dim, num_heads)
    self.ffn = point_wise_feed_forward_network(mlp_dim, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    #self.dropout1 = tf.keras.layers.Dropout(rate)
    #self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)
    out1 = self.layernorm1(x + attn_output)
    ffn_output = self.ffn(out1)
    out2 = self.layernorm2(out1 + ffn_output)

    return out2

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, mlp_dim, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.mlp_dim = mlp_dim

    assert mlp_dim % self.num_heads == 0

    self.depth = mlp_dim // self.num_heads

    self.wq = tf.keras.layers.Dense(mlp_dim)
    self.wk = tf.keras.layers.Dense(mlp_dim)
    self.wv = tf.keras.layers.Dense(mlp_dim)

    self.dense = tf.keras.layers.Dense(mlp_dim)

  def split_heads(self, x, batch_size):
    """最後の次元を(num_heads, depth)に分割。
    結果をshapeが(batch_size, num_heads, seq_len, depth)となるようにリシェイプする。
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)

    q = self.split_heads(q, batch_size)
    k = self.split_heads(k, batch_size)
    v = self.split_heads(v, batch_size)

    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.mlp_dim))

    output = self.dense(concat_attention)

    return output, attention_weights