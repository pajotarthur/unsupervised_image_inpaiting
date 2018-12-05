import numpy as np

from src.utils.metrics import Metrics


np.random.seed(1234)


def random_data_generator():
    """ fake data generator
    """
    n_batches = np.random.randint(1, 10)
    for _ in range(n_batches):
        batch_size = np.random.randint(1, 5)
        data = np.random.normal(size=(batch_size, 3))
        target = np.random.randint(10, size=batch_size)
        yield (data, target)

def training_data():
    return random_data_generator()


def validation_data():
    return random_data_generator()


def test_data():
    return random_data_generator()


def oracle(data, target):
    """ fake metric data generator
    """
    loss = np.random.rand()
    acc1 = np.random.rand() + 70
    acck = np.random.rand() + 90

    return loss, acc1, acck


#----------------------------------------------------------
# Prepare logging
#----------------------------------------------------------
# create Emeriment
n_epochs = 10
m = Metrics(time_indexing=False, xlabel='Epoch')

# create parent metric for training metrics (easier interface)
m.Parent(name='train',
         children=(m.AvgMetric(name='loss'),
                   m.AvgMetric(name='acc1'),
                   m.AvgMetric(name='acck')))
# same for validation metrics (note all children inherit parent from parent)
m.Parent(name='val',
         children=(m.AvgMetric(name='loss'),
                   m.AvgMetric(name='acc1'),
                   m.AvgMetric(name='acck')))
bestl = m.BestMetric(name="best_train_loss")
bestk = m.BestMetric(name="acck")

#----------------------------------------------------------
# Training
#----------------------------------------------------------

for epoch in range(n_epochs):
    # train model
    for (x, y) in training_data():
        loss, acc1, acck = oracle(x, y)
        # accumulate metrics (average over mini-batches)
        m.train.update(loss=loss, acc1=acc1,
                       acck=acck, n=len(x))
    bestl.update(m.train.loss.value)  # will update only if better than previous values

    m.train.reset()

    for (x, y) in validation_data():
        loss, acc1, acck = oracle(x, y)
        m.val.update(loss=loss, acc1=acc1,
                     acck=acck, n=len(x))
    bestk.update(m.val.acck.value)  # will update only if better than previous values

print("=" * 50)
print("Best Performance On Validation Data:")
print("-" * 50)
print("Prec@1: \t {0:.4f}".format(bestl.value))
print("Prec@k: \t {0:.2f}%".format(bestk.value))
print("=" * 50)        