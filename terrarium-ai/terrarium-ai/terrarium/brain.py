"""Batched recurrent brains. Every animal owns its own weights, stored as one row of a [N, P] tensor,
so a whole population thinks and learns in a few tensor operations (and runs on a GPU)."""
import math
import torch


class BrainSpec:
    def __init__(self, n_in: int, hidden: int, n_out: int):
        self.I, self.H, self.O = n_in, hidden, n_out
        self.layout = [("Wx", (n_in, hidden)), ("Wh", (hidden, hidden)), ("bh", (hidden,)),
                       ("Wo", (hidden, n_out)), ("bo", (n_out,))]
        self.P = sum(math.prod(s) for _, s in self.layout)

    def unpack(self, params):
        n, o, out = params.shape[0], 0, {}
        for name, shp in self.layout:
            k = math.prod(shp)
            out[name] = params[:, o:o + k].reshape(n, *shp)
            o += k
        return out

    def init(self, n, device):
        p = torch.randn(n, self.P, device=device) * 0.4
        p[:, self.P - self.O + 1] += 0.5      # lean toward moving
        p[:, self.P - self.O + 2] -= 0.3      # lean toward not biting
        return p

    def forward(self, params, x, h):
        """x: [N, I] senses, h: [N, H] memory. Returns actions [N, O] in (-1, 1) and new memory."""
        w = self.unpack(params)
        pre = torch.einsum("ni,nih->nh", x, w["Wx"]) + torch.einsum("nh,nhk->nk", h, w["Wh"]) + w["bh"]
        h2 = torch.tanh(pre)
        out = torch.tanh(torch.einsum("nh,nho->no", h2, w["Wo"]) + w["bo"])
        return out, h2
