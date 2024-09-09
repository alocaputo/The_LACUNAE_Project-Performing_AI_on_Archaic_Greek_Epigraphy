import os
import json
import argparse
from tqdm import tqdm

import functools
import pickle

from absl import app
from absl import flags
from ithaca.eval import inference
from ithaca.models.model import Model
from ithaca.util.alphabet import GreekAlphabet
import jax

def load_checkpoint(path):
  """Loads a checkpoint pickle.

  Args:
    path: path to checkpoint pickle

  Returns:
    a model config dictionary (arguments to the model's constructor), a dict of
    dicts containing region mapping information, a GreekAlphabet instance with
    indices and words populated from the checkpoint, a dict of Jax arrays
    `params`, and a `forward` function.
  """

  # Pickled checkpoint dict containing params and various config:
  with open(path, 'rb') as f:
    checkpoint = pickle.load(f)

  # We reconstruct the model using the same arguments as during training, which
  # are saved as a dict in the "model_config" key, and construct a `forward`
  # function of the form required by attribute() and restore().
  params = jax.device_put(checkpoint['params'])
  model = Model(**checkpoint['model_config'])
  forward = functools.partial(model.apply, params)

  # Contains the mapping between region IDs and names:
  region_map = checkpoint['region_map']

  # Use vocabulary mapping from the checkpoint, the rest of the values in the
  # class are fixed and constant e.g. the padding symbol
  alphabet = GreekAlphabet()
  alphabet.idx2word = checkpoint['alphabet']['idx2word']
  alphabet.word2idx = checkpoint['alphabet']['word2idx']

  return checkpoint['model_config'], region_map, alphabet, params, forward

def restore_inscriptions(results_path, cuda_id, start_idx, end_idx):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    test_data = json.load(open('data/archaic/test_common.json'))

    (model_config, region_map, alphabet, params,
      forward) = load_checkpoint('../ithaca/checkpoint.pkl')
    vocab_char_size = model_config['vocab_char_size']
    vocab_word_size = model_config['vocab_word_size']

    for inscription in tqdm(test_data[start_idx:end_idx]):
        input_text = inscription['masked_ithaca']
        phi_id = inscription['id']

        if not 50 <= len(input_text) <= 750:
            print("mmmmhhhh")
            continue
            raise ValueError(
                f'Text should be between 50 and 750 chars long, but the input was '
                f'{len(input_text)} characters')

        json_path = os.path.join(results_path, f'{phi_id}.json')

        if os.path.isfile(json_path): # Do not restore already restored inscriptions 
            continue

        restoration = inference.restore(
            input_text,
            forward=forward,
            params=params,
            alphabet=alphabet,
            vocab_char_size=vocab_char_size,
            vocab_word_size=vocab_word_size)

        with open(json_path, 'w') as f:
            f.write(restoration.json(indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore inscriptions")
    parser.add_argument("--results_path", type=str, default='results/archaic/ithaca', help="Path to save results")
    parser.add_argument("--cuda", type=str, default='0', help="CUDA device [0,1,2]")
    parser.add_argument("--start", type=int, default='0', help="Index start")
    parser.add_argument("--end", type=int, default='95', help="Index start")
    args = parser.parse_args()

    restore_inscriptions(args.results_path, args.cuda, args.start, args.end)
