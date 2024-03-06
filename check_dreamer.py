import dreamerv3.embodied
import pprint
import jax
import jax.numpy as jnp

filename = dreamerv3.embodied.path.Path('./../dreamerv3_offline/logdir/hopper-medium-v2/checkpoint.ckpt')
ckpt = dreamerv3.embodied.basics.unpack(filename.read('rb'))['agent']
ckpt = {k: v for (k, v) in ckpt.items() if 'rssm' in k}
pprint.pprint(jax.tree_util.tree_map(jnp.shape, ckpt), width=1)
