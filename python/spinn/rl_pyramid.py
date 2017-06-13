
import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from spinn.util.blocks import Embed, to_gpu, MLP, Linear, HeKaimingInitializer
from spinn.util.misc import Args, Vocab
from spinn.util.blocks import SimpleTreeLSTM
from spinn.util.sparks import sparks

def build_model(data_manager, initial_embeddings, vocab_size,
                num_classes, FLAGS, context_args, composition_args):
    use_sentence_pair = data_manager.SENTENCE_PAIR_DATA
    model_cls = RLPyramid

    return model_cls(model_dim=FLAGS.model_dim,
                     word_embedding_dim=FLAGS.word_embedding_dim,
                     vocab_size=vocab_size,
                     initial_embeddings=initial_embeddings,
                     num_classes=num_classes,
                     embedding_keep_rate=FLAGS.embedding_keep_rate,
                     use_sentence_pair=use_sentence_pair,
                     use_difference_feature=FLAGS.use_difference_feature,
                     use_product_feature=FLAGS.use_product_feature,
                     classifier_keep_rate=FLAGS.semantic_classifier_keep_rate,
                     mlp_dim=FLAGS.mlp_dim,
                     num_mlp_layers=FLAGS.num_mlp_layers,
                     mlp_ln=FLAGS.mlp_ln,
                     context_args=context_args,
                     gated=FLAGS.pyramid_gated,
                     selection_keep_rate=FLAGS.pyramid_selection_keep_rate,
                     )


class RLPyramid(nn.Module):

    def __init__(self, model_dim=None,
                 word_embedding_dim=None,
                 vocab_size=None,
                 initial_embeddings=None,
                 num_classes=None,
                 embedding_keep_rate=None,
                 use_sentence_pair=False,
                 classifier_keep_rate=None,
                 mlp_dim=None,
                 num_mlp_layers=None,
                 mlp_ln=None,
                 context_args=None,
                 gated=None,
                 selection_keep_rate=None,
                 pyramid_selection_keep_rate=None,
                 reinforce=None, # RL model
                 **kwargs
                 ):
        super(RLPyramid, self).__init__()

        self.use_sentence_pair = use_sentence_pair
        self.model_dim = model_dim
        self.gated = gated
        self.selection_keep_rate = selection_keep_rate

        classifier_dropout_rate = 1. - classifier_keep_rate

        args = Args()
        args.size = model_dim
        args.input_dropout_rate = 1. - embedding_keep_rate

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        self.embed = Embed(word_embedding_dim, vocab.size, vectors=vocab.vectors)

        self.composition_fn = SimpleTreeLSTM(model_dim / 2,
                                             composition_ln=False)
        self.selection_fn = Linear(initializer=HeKaimingInitializer)(model_dim, 1)

        # TODO: Set up layer norm.

        mlp_input_dim = model_dim * 2 if use_sentence_pair else model_dim

        self.mlp = MLP(mlp_input_dim, mlp_dim, num_classes,
                       num_mlp_layers, mlp_ln, classifier_dropout_rate)

        self.encode = context_args.encoder
        self.reshape_input = context_args.reshape_input
        self.reshape_context = context_args.reshape_context

    def run_pyramid(self, x, show_sample=False):
        batch_size, seq_len, model_dim = x.data.size()

        # Each item is a list of B x D phrase vectors for a given layer.
        phrase_vectors = []
        phrase_vectors.append(torch.chunk(x, seq_len, 1))

        # Temp fix:
        show_sample = False

        if show_sample:
            print

        #left_ids = range(seq_len - 1)
        #right_ids = [x+1 for x in left_ids]
        # TODO: fill in the rest of the faster model.

        # composition_results across layers:
        # | | | |_/ / /    choices: 0-5 len: 7
        # | | | |__/ /              0-4 len: 6
        # | |_/ / __/               0-3
        # |_/ _/ /
        # |__/ _/
        # |___/
        # |  -> output

        for layer in range(seq_len - 1, 0, -1):
            # Composition_results need to be recalculated to support
            # batching multiple examples together.
            composition_results = []
            selection_logits_list = []

            for position in range(layer):
                left = torch.squeeze(phrase_vectors[-1][position])
                right = torch.squeeze(phrase_vectors[-1][position + 1])
                composition_results.append(self.composition_fn(left, right))

            # selection_logits: FT B x len(comp_results)
            for position in range(layer):    
                selection_logits_list.append(self.selection_fn(composition_results[position]))
            selection_logits = torch.cat(selection_logits_list, 1)
            selection_probs = F.softmax(selection_logits) # FT, B x len(comp_results)

            if show_sample:
                print sparks(np.transpose(selection_probs[0,:].data.cpu().numpy()).tolist())

            # "dropout" some candidate selections.
            if self.training and self.selection_keep_rate is not None:
                # 0 = drop, 1 = keep
                keep_prob = to_gpu(torch.ones(1, 1)) * self.selection_keep_rate
                noise = torch.bernoulli(keep_prob.expand_as(selection_logits))
                selection_probs *= Variable(noise)

            # RL happens here. Change selection_probs to a hard decision on
            # which one to compose. Remember logits and selections.
            # TODO: implement
            selected = tensor # LT, B

            layer_state_pairs = [] # State of sentence in the next step
            for position in range(layer):
                # copy_left: if choose 5+6, copy 0-4
                # copy_left: if choose 5+6, copy 7-S
                copy_left = position.lt(selected)
                copy_right = position.gt(selected + 1)
                select = (1 - copy_left) * (1 - copy_right)
                # These three are mutually exclusive.
                # Assert: copy_left + copy_right + select = 1 everywhere
                # Assert: copy_left, copy_right, select >= 0

                left = torch.squeeze(phrase_vectors[-1][position])
                right = torch.squeeze(phrase_vectors[-1][position + 1])
                composition_result = composition_results[position]
                new_state_pair = copy_left.expand_as(left) * left \
                    + copy_right.expand_as(right) * right \
                    + select.unsqueeze(1).expand_as(composition_result) * composition_result
                layer_state_pairs.append(new_state_pair)

            phrase_vectors.append(layer_state_pairs)

        # TODO: return choices too
        return phrase_vectors[-1][-1]

    def run_embed(self, x):
        batch_size, seq_length = x.size()
        print('x.size', x.size())

        embeds = self.embed(x) # FT, B x S x W
        embeds = self.reshape_input(embeds, batch_size, seq_length)
        embeds = self.encode(embeds)
        embeds = self.reshape_context(embeds, batch_size, seq_length)
        embeds = torch.cat([b.unsqueeze(0) for b in torch.chunk(embeds, batch_size, 0)], 0)
        # TODO: remove
        print('embeds.size', embeds.size())

        return embeds

    def forward(self, sentences, transitions=None, y_batch=None, show_sample=False):
        '''
        sentences: Tensor
        transitions: Tensor
        y_batch: Tensor
        '''
        # Useful when investigating dynamic batching:
        # self.seq_lengths = sentences.shape[1] - (sentences == 0).sum(1)

        x = self.unwrap(sentences, transitions) # LT, B x S
        emb = self.run_embed(x)
        hh = self.run_pyramid(emb, show_sample)
        h = self.wrap(hh)
        output = self.mlp(h)

        probs = F.softmax(output).data.cpu() # FT, B x V
        target = torch.from_numpy(y_batch).long() # LT, B x 1

        # Get Reward.
        rewards = self.build_reward(probs, target) # FT, B x 1
        baseline = self.build_baseline(rewards, sentences, transitions, y_batch)
        advantage = self.advantage(rewards, baseline) # FT, B x 1
        policy_loss = self.reinforce(advantage)

        return output

    # --- Sentence Style Switches ---

    def unwrap(self, sentences, transitions):
        if self.use_sentence_pair:
            return self.unwrap_sentence_pair(sentences, transitions)
        return self.unwrap_sentence(sentences, transitions)

    def wrap(self, hh):
        if self.use_sentence_pair:
            return self.wrap_sentence_pair(hh)
        return self.wrap_sentence(hh)

    # --- Sentence Specific ---

    def unwrap_sentence_pair(self, sentences, transitions):
        x_prem = sentences[:, :, 0]
        x_hyp = sentences[:, :, 1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        return to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))

    def wrap_sentence_pair(self, hh):
        batch_size = hh.size(0) / 2
        h = torch.cat([hh[:batch_size], hh[batch_size:]], 1)
        return h

    # --- Sentence Pair Specific ---

    def unwrap_sentence(self, sentences, transitions):
        return to_gpu(Variable(torch.from_numpy(sentences), volatile=not self.training))

    def wrap_sentence(self, hh):
        return hh
