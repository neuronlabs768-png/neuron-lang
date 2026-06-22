// transformer.nr — A simple Transformer model in NEURON
// Demonstrates: model/layer declarations, Tensor shapes, Learnable, @compile

@compile(target="auto")
layer MultiHeadAttention(d_model: Int, n_heads: Int):
  head_dim: Int = 64
  wq: Tensor[d_model, d_model] = glorot(d_model, d_model)
  wk: Tensor[d_model, d_model] = glorot(d_model, d_model)
  wv: Tensor[d_model, d_model] = glorot(d_model, d_model)
  wo: Tensor[d_model, d_model] = glorot(d_model, d_model)

  fn forward(self, x: Tensor[B, S, d_model]) -> Tensor[B, S, d_model]:
    let q = x @ self.wq
    let k = x @ self.wk
    let v = x @ self.wv
    let scores = q @ k
    let attn = softmax(scores)
    let out = attn @ v
    return out @ self.wo

layer FeedForward(d_model: Int):
  w1: Tensor[d_model, 2048] = glorot(d_model, 2048)
  w2: Tensor[2048, d_model] = glorot(2048, d_model)

  fn forward(self, x: Tensor[B, S, d_model]) -> Tensor[B, S, d_model]:
    let h = gelu(x @ self.w1)
    return h @ self.w2

@compile(target="auto")
model Transformer(d_model: Int, n_heads: Int, n_layers: Int):
  embed: Tensor[50000, d_model] = glorot(50000, d_model)

  fn forward(self, tokens: Tensor[B, S]) -> Tensor[B, S, 50000]:
    let x = tokens
    let logits = x @ self.embed
    return softmax(logits)

  fn train(self, data: Dataset) [Effect[Mut[self], IO]]:
    for batch in data:
      let loss = cross_entropy(self.forward(batch), batch)
      update self by adam(grad(loss), lr=3e-4)
