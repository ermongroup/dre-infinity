"""Training and evaluation"""

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import tensorflow as tf
import wandb

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.DEFINE_string("doc", None, "exp_name")
flags.DEFINE_bool("toy", False, "whether to run toy experiment")
flags.DEFINE_bool("flow", False, "whether to encode the data into latent space using a pre-trained flow")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
  mode = None

  if FLAGS.mode == 'eval':
    mode = 'disabled'
  # TODO: set up wandb and replace names here
  wandb.init(
    project='your-project-here',
    entity='your-project-here',
    name=FLAGS.doc,
    mode=mode
  )
  
  if FLAGS.mode == "train":
    # Create the working directory
    tf.io.gfile.makedirs(FLAGS.workdir)
    # Set logger so that it outputs to both console and file
    # Make logging work for both disk and Google Cloud Storage
    gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

    # save config
    print(FLAGS.config)
    with open(os.path.join(FLAGS.workdir, 'config.txt'), 'w') as f:
      print(FLAGS.config, file=f)

      # Run the training pipeline
      if not FLAGS.toy:
        if not FLAGS.flow:
          import run_lib
          print('running normal score matching method without any flows!')
          run_lib.train(FLAGS.config, FLAGS.workdir)
        else:
          logger.info("Leveraging pre-trained flow model for score estimation!")
          if 'rq_nsf' in FLAGS.config.training.z_space_model:
            import run_lib_rqnsf_flow
            # TODO (HACK): preprocessing for each type of flow is annoying, but eventually want to merge these two files
            print('running rq_nsf-specific training code')
            run_lib_rqnsf_flow.train(FLAGS.config, FLAGS.workdir)
          else:
            import run_lib_flow
            run_lib_flow.train(FLAGS.config, FLAGS.workdir)
      else:
        import toy_run_lib
        toy_run_lib.train(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == "eval":
    # Run the evaluation pipeline
    if FLAGS.flow:
      import run_lib_rqnsf_flow
      if 'rq_nsf' in FLAGS.config.training.z_space_model:
        run_lib_rqnsf_flow.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
      else:
        import run_lib_flow
        run_lib_flow.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
    else:
      import run_lib
      run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
