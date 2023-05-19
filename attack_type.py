from attacks import *
import numpy as np

attackers = {
    'fgsm_delta': lambda loss_fn, eps, nb_iter, eps_iter, num_classes, initial_const: FGSM_delta(loss_fn, num_classes, eps),
    'fgsm': lambda predict, loss_fn, eps, nb_iter, eps_iter, num_classes, initial_const: FGSM(predict, loss_fn, eps),
    'bim': lambda predict, loss_fn, eps, nb_iter, eps_iter, num_classes, initial_const: LinfBasicIterativeAttack(predict, loss_fn, eps, nb_iter, eps_iter),
    'mi_fgsm': lambda predict, loss_fn, eps, nb_iter, eps_iter, num_classes, initial_const: MomentumIterativeAttack(predict, loss_fn, eps, nb_iter, eps_iter,\
                                                        decay_factor=1., ord=np.inf),
    'pgd': lambda predict, loss_fn, eps, nb_iter, eps_iter, num_classes, initial_const: PGDAttack(predict, loss_fn, eps, nb_iter, eps_iter, rand_init=True,\
                                                        ord=np.inf, l1_sparsity=None),
    'c&w': lambda predict, loss_fn, eps, nb_iter, eps_iter, num_classes, initial_const: CarliniWagnerL2Attack(predict, num_classes=num_classes, confidence=0, targeted=False, learning_rate=0.01,
                                                        binary_search_steps=9, max_iterations=1000, abort_early=True, initial_const=1e-3, loss_fn=None),
    'ddn': lambda predict, loss_fn, eps, nb_iter, eps_iter, num_classes, initial_const: DDNL2Attack(predict,nb_iter=100, gamma=0.05, init_norm=1., quantize=True,\
                                                        levels=256, clip_min=0., clip_max=1., targeted=False, loss_fn=None),
}