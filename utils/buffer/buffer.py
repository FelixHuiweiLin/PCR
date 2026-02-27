from utils.setup_elements import input_size_match, n_classes
from utils import name_match #import update_methods, retrieve_methods
from utils.utils import maybe_cuda
import torch
from utils.buffer.buffer_utils import BufferClassTracker
from utils.setup_elements import n_classes

class Buffer(torch.nn.Module):
    def __init__(self, model, params):
        super().__init__()
        self.params = params
        self.model = model
        self.cuda = self.params.cuda
        self.current_index = 0
        self.n_seen_so_far = 0
        self.device = "cuda" if self.params.cuda else "cpu"
        self.samples_per_cls = torch.ones([n_classes[params.data]])
        # define buffer
        buffer_size = params.mem_size
        print('buffer has %d slots' % buffer_size)
        input_size = input_size_match[params.data]
        output_size = n_classes[params.data]
        buffer_img = maybe_cuda(torch.FloatTensor(buffer_size, *input_size).fill_(0))
        buffer_label = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))
        buffer_logits = maybe_cuda(torch.FloatTensor(buffer_size, output_size).fill_(0))
        buffer_age = maybe_cuda(torch.LongTensor(buffer_size).fill_(0))
        if self.params.data in ['tinyimagenet', 'mini_imagenet']:
            buffer_features = maybe_cuda(torch.FloatTensor(buffer_size, 640).fill_(0))
        else:
            buffer_features = maybe_cuda(torch.FloatTensor(buffer_size, 160).fill_(0))

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('buffer_img', buffer_img)
        self.register_buffer('buffer_label', buffer_label)
        self.register_buffer('buffer_logits', buffer_logits)
        self.register_buffer('buffer_age', buffer_age)
        self.register_buffer('buffer_features', buffer_features)

        # define update and retrieve method
        self.update_method = name_match.update_methods[params.update](params)
        self.retrieve_method = name_match.retrieve_methods[params.retrieve](params)

        if self.params.buffer_tracker:
            self.buffer_tracker = BufferClassTracker(n_classes[params.data], self.device)

    def update(self, x, y, logits, features, **kwargs):
        return self.update_method.update(buffer=self, x=x, y=y, logits=logits, features=features, **kwargs)


    def retrieve(self, **kwargs):
        return self.retrieve_method.retrieve(buffer=self, **kwargs)

    def update_gmed(self, mem_x, mem_logits, mem_age, mem_indices):
        self.buffer_img[mem_indices] = mem_x
        self.buffer_logits[mem_indices] = mem_logits
        self.buffer_age[mem_indices] = mem_age

    def sample_pos_neg(self, in_x, in_y):
        x = in_x
        label = in_y

        bx = torch.cat([self.buffer_img[:self.current_index], x])
        by = torch.cat([self.buffer_label[:self.current_index], label])
        bidx = torch.arange(bx.size(0)).to(bx.device)

        same_label = label.view(1, -1) == by.view(-1, 1)
        same_ex = bidx[-x.size(0):].view(1, -1) == bidx.view(-1, 1)
        task_labels = label.unique()

        valid_pos = same_label
        valid_neg = ~same_label
        has_valid_pos = valid_pos.sum(0) > 0
        has_valid_neg = valid_neg.sum(0) > 0
        invalid_idx = ~has_valid_pos | ~has_valid_neg
        if invalid_idx.any():
            # so the fetching operation won't fail
            valid_pos[:, invalid_idx] = 1
            valid_neg[:, invalid_idx] = 1

        is_invalid = torch.zeros_like(label).bool()
        is_invalid[invalid_idx] = 1

        # fetch positive samples
        pos_idx = torch.multinomial(valid_pos.float().T, 1).squeeze(1)
        neg_idx = torch.multinomial(valid_neg.float().T, 1).squeeze(1)
        n_fwd = torch.stack((bidx[-x.size(0):], pos_idx, neg_idx), 1)[~invalid_idx].unique().size(0)

        return bx[pos_idx], \
               bx[neg_idx], \
               by[pos_idx], \
               by[neg_idx], \
               is_invalid, \
               n_fwd
