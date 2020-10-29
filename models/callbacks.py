from keras.callbacks import Callback, CSVLogger
import collections
import csv
import six
import numpy as np
from tensorflow.python.util.compat import collections_abc


class BatchCSVLogger(CSVLogger):
    def __init__(self, *args, X=None, Y=None, **kwargs):
        self.global_batch_count = 0
        self.X = X
        self.Y = Y
        super(BatchCSVLogger, self).__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        self.global_batch_count += 1
        if self.global_batch_count % 100 == 0:
            batch = self.global_batch_count
            logs = logs or {}

            def handle_value(k):
                is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
                if isinstance(k, six.string_types):
                    return k
                elif isinstance(k, collections_abc.Iterable) and not is_zero_dim_ndarray:
                    return '"[%s]"' % (', '.join(map(str, k)))
                else:
                    return k

            if self.keys is None:
                self.keys = sorted(logs.keys())

            if self.model.stop_training:
                # We set NA so that csv parsers do not fail for this last epoch.
                logs = dict((k, logs[k]) if k in logs else (k, 'NA') for k in self.keys)

            if not self.writer:

                class CustomDialect(csv.excel):
                    delimiter = self.sep

                fieldnames = ['batch'] + self.keys + ["val_acc", "val_loss", "val_true_acc"]
                if six.PY2:
                    fieldnames = [unicode(x) for x in fieldnames]

                self.writer = csv.DictWriter(
                    self.csv_file,
                    fieldnames=fieldnames,
                    dialect=CustomDialect)
                if self.append_header:
                    self.writer.writeheader()

            row_dict = collections.OrderedDict({'batch': self.global_batch_count})
            row_dict.update((key, handle_value(logs[key])) for key in self.keys)
            row_dict.update([("batch", self.global_batch_count)])
            # predict model
            predictions = self.model.evaluate(self.X, self.Y)
            for metric_name, prediction in zip(self.model.metrics_names, predictions):
                row_dict.update([("val_{}".format(metric_name), prediction)])

            self.writer.writerow(row_dict)
            self.csv_file.flush()

