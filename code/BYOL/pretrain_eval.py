import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import argparse
import numpy as np
import tensorflow as tf

from datasets import CIFAR10
from models import ResNet18, ResNet34, ProjectionHead
from losses import byol_loss
from linearevaluation import compute_test_accuracy

from models import ResNet18, ResNet34, ClassificationHead
from sklearn.metrics import top_k_accuracy_score




encoders = {'resnet18': ResNet18, 'resnet34': ResNet34}


def perform_evaluation(args, weights):
    # Load CIFAR-10 dataset
    data = CIFAR10()

    # Define hyperparameters
    num_epochs = 10
    batch_size = args.batch_size

    # Instantiate networks f and c
    f_net = encoders[args.encoder]()
    c_net = ClassificationHead()

    # Initialize the weights of f and c
    x, y = data.get_batch_finetuning(batch_id=0, batch_size=batch_size)
    h = f_net(x, training=False)
    print('Shape of h:', h.shape)
    s = c_net(h)
    print('Shape of s:', s.shape)

    # Load the weights of f from pretraining
    f_net.set_weights(np.copy(weights))
    print('Weights of f loaded.')


    # Define optimizer
    batches_per_epoch = data.num_train_images // batch_size
    total_update_steps = num_epochs * batches_per_epoch
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(5e-2, total_update_steps, 5e-4, power=2)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    
    @tf.function
    def train_step_evaluation(x, y):  # (bs, 32, 32, 3), (bs)

        # Forward pass
        with tf.GradientTape() as tape:
            h = f_net(x, training=False)  # (bs, 512)
            y_pred_logits = c_net(h)  # (bs, 10)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_pred_logits))
        
        # Backward pass
        grads = tape.gradient(loss, c_net.trainable_variables)
        opt.apply_gradients(zip(grads, c_net.trainable_variables))

        return loss


    print("Finetuning linear head")
    # Fine Tune linear head for current weights
    for epoch_id in range(num_epochs):
        print("Epoch {}/10...".format(epoch_id))
        data.shuffle_training_data()
        
        for batch_id in range(batches_per_epoch):
	        x, y = data.get_batch_finetuning(batch_id, batch_size)
	        loss = train_step_evaluation(x, y)
            #if (batch_id + 1) % log_every == 0:
    
    # Compute classification accuracy on test set
    test_accuracy = compute_test_accuracy(data, f_net, c_net)
    test_top5_accuracy = compute_top5_test_accuracy(data,f_net,c_net)

    return test_accuracy,test_top5_accuracy


def main(args):

    print('Going to load dataset')
    # Load CIFAR-10 dataset
    data = CIFAR10()

    print('Loaded')
    # Instantiate networks
    f_online = encoders[args.encoder]()
    g_online = ProjectionHead()
    q_online = ProjectionHead()

    f_target = encoders[args.encoder]()
    g_target = ProjectionHead()


    # Initialize the weights of the networks
    x = tf.random.normal((256, 32, 32, 3))
    h = f_online(x, training=False)
    print('Initializing online networks...')
    print('Shape of h:', h.shape)
    z = g_online(h, training=False)
    print('Shape of z:', z.shape)
    p = q_online(z, training=False)
    print('Shape of p:', p.shape)

    h = f_target(x, training=False)
    print('Initializing target networks...')
    print('Shape of h:', h.shape)
    z = g_target(h, training=False)
    print('Shape of z:', z.shape)
    
    num_params_f = tf.reduce_sum([tf.reduce_prod(var.shape) for var in f_online.trainable_variables])    
    print('The encoders have {} trainable parameters each.'.format(num_params_f))


    # Define optimizer
    lr = 1e-3 * args.batch_size / 512
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    print('Using Adam optimizer with learning rate {}.'.format(lr))


    @tf.function
    def train_step_pretraining(x1, x2):  # (bs, 32, 32, 3), (bs, 32, 32, 3)

        # Forward pass
        h_target_1 = f_target(x1, training=True)
        z_target_1 = g_target(h_target_1, training=True)

        h_target_2 = f_target(x2, training=True)
        z_target_2 = g_target(h_target_2, training=True)

        with tf.GradientTape(persistent=True) as tape:
            h_online_1 = f_online(x1, training=True)
            z_online_1 = g_online(h_online_1, training=True)
            p_online_1 = q_online(z_online_1, training=True)
            
            h_online_2 = f_online(x2, training=True)
            z_online_2 = g_online(h_online_2, training=True)
            p_online_2 = q_online(z_online_2, training=True)
            
            p_online = tf.concat([p_online_1, p_online_2], axis=0)
            z_target = tf.concat([z_target_2, z_target_1], axis=0)
            loss = byol_loss(p_online, z_target)



        # Backward pass (update online networks)
        grads = tape.gradient(loss, f_online.trainable_variables)
        opt.apply_gradients(zip(grads, f_online.trainable_variables))
        grads = tape.gradient(loss, g_online.trainable_variables)
        opt.apply_gradients(zip(grads, g_online.trainable_variables))
        grads = tape.gradient(loss, q_online.trainable_variables)
        opt.apply_gradients(zip(grads, q_online.trainable_variables))
        del tape

        return loss


    batches_per_epoch = data.num_train_images // args.batch_size
    log_every = 10  # batches
    save_every = 100  # epochs

    losses = []
    f_target_weights = []
    for epoch_id in range(args.num_epochs):
        data.shuffle_training_data()
        
        for batch_id in range(batches_per_epoch):
            x1, x2 = data.get_batch_pretraining(batch_id, args.batch_size)
            loss = train_step_pretraining(x1, x2)
            losses.append(float(loss))
            
            # Update target networks (exponential moving average of online networks)
            beta = 0.99

            f_target_weights = f_target.get_weights()
            f_online_weights = f_online.get_weights()
            for i in range(len(f_online_weights)):
                f_target_weights[i] = beta * f_target_weights[i] + (1 - beta) * f_online_weights[i]
            f_target.set_weights(f_target_weights)
            
            g_target_weights = g_target.get_weights()
            g_online_weights = g_online.get_weights()
            for i in range(len(g_online_weights)):
                g_target_weights[i] = beta * g_target_weights[i] + (1 - beta) * g_online_weights[i]
            g_target.set_weights(g_target_weights)

            if (batch_id + 1) % log_every == 0:
                print('[Epoch {}/{} Batch {}/{}] Loss={:.5f}.'.format(epoch_id+1, args.num_epochs, batch_id+1, batches_per_epoch, loss))

	# Every 10 epochs compute test
        if (epoch_id + 1) % save_every == 0:
	    f_online.save_weights('f_online_{}.h5'.format(epoch_id + 1))
	    print('Weights of f saved.')
	    # Evaluate in epoch 
	    acc,top_5_acc = perform_evaluation(args,f_target_weights)
        
	    # Add tensorflow saving
	    train_summary_writer = tf.summary.create_file_writer(args.log_dir)
	    with train_summary_writer.as_default():
		tf.summary.scalar('top_1_acc',float(acc),step=epoch)
		tf.summary.scalar('top_5_acc',float(top_5_acc),step=epoch)

        
        # Add tensorflow saving for loss in each epoch
        train_summary_writer = tf.summary.create_file_writer(args.log_dir)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', float(losses[-1]), step=epoch)
            


        
    
    #np.savetxt('losses.txt', tf.stack(losses).numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--encoder', type=str, required=True, choices=['resnet18', 'resnet34'], help='Encoder architecture')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for pretraining')
    parser.add_argument('--logdir', type=str,required = True, help='Directory to store results')
    args = parser.parse_args()
    main(args)
