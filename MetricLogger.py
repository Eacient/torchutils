import torch
import torch.distributed as dist
import time
import datetime
from collections import defaultdict, deque
from Distributed import is_dist_avail_and_initialized

class SmoothedValue(object):
    """
    keep a series of values and provide access to 
    - smoothed values over a window
    - global average

    implement __str__ to expose access to calculated properties
    usage
    init(fmt=) before training and be added to a metric logger
        fmt can contain variables including
        - median
        - max
        - avg(window avg)
        - global_avg
        - value
        more supported variable can be add by adding the corresponding
        calculating funcrion and modify the __str__ implementation
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, batch_mean, batch_size=1):
        self.deque.append(batch_mean)
        self.count += batch_size
        self.total += batch_mean * batch_size
    
    def synchronize_between_process(self):
        """
        warning: only synchornize base val count and total
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()
    
    @property
    def global_avg(self):
        return self.total / self.count
    
    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]
    
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg = self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    """
    composed of multiple named meters
    and support log through their __str__ implementation
    main usage:
    - add_meter() before training
    - update() to save within training
    - log_every() to wrap the dataloader loop
    """

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def __getattr__(self, attr):
        # enable dict style access to meters
        # self.meter_name == self.meters[meter_name]
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v.torch.Tensor):
                v = v.item()
            assert isinstance(v, (float,int))
            self.meters[k].update(v)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def log_every(self, iterable, print_freq, header=None):
        """
    serve as an iterator wrapper for dataloader or other iterables
    automatically cal 
    - data_time: at the begin of iteration
    - iter_time at the begin of next iteration
        """
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)

        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

