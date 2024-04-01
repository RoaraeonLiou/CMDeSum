import torch


class BatchNode(object):
    def __init__(self, list_node):
        self.list_node = list_node

    def get_comment(self):
        comment_list = [node.commentID for node in self.list_node]
        return torch.cat(comment_list, dim=0)

    def get_dec_state(self):
        dec_state_list = [node.dec_state for node in self.list_node]
        batch_dec_state = [torch.cat([batch_state[0] for batch_state in dec_state_list], dim=0),
                           torch.cat([batch_state[1] for batch_state in dec_state_list], dim=0)]
        if dec_state_list[0][2][0] is None:
            batch_dec_state.append(dec_state_list[0][2])
        else:
            state_3 = []
            for i in range(len(dec_state_list[0][2])):
                state_3.append(torch.cat([batch_state[2][i] for batch_state in dec_state_list], dim=0))
            assert len(state_3) == len(dec_state_list[0][2])
            batch_dec_state.append(state_3)
        return batch_dec_state

    def if_allEOS(self, eos_token):
        for node in self.list_node:
            if node.history_word[-1] != eos_token:
                return False
        return True


class BatchNodeWithKeywords(object):
    def __init__(self, list_node):
        self.list_node = list_node

    def get_comment(self):
        comment_list = [node.commentID for node in self.list_node]
        return torch.cat(comment_list, dim=0)

    def get_dec_state(self):
        dec_state_list = [node.dec_state for node in self.list_node]
        try:
            batch_dec_state = [torch.cat([batch_state[0] for batch_state in dec_state_list], dim=0),
                               torch.cat([batch_state[1] for batch_state in dec_state_list], dim=0),
                               torch.cat([batch_state[2] for batch_state in dec_state_list], dim=0),
                               torch.cat([batch_state[3] for batch_state in dec_state_list], dim=0),
                               torch.cat([batch_state[4] for batch_state in dec_state_list], dim=0),
                               torch.cat([batch_state[5] for batch_state in dec_state_list], dim=0)]
        except Exception as e:
            batch_dec_state = [torch.cat([batch_state[0] for batch_state in dec_state_list], dim=0),
                               torch.cat([batch_state[1] for batch_state in dec_state_list], dim=0),
                               torch.cat([batch_state[2] for batch_state in dec_state_list], dim=0),
                               torch.cat([batch_state[3] for batch_state in dec_state_list], dim=0),
                               torch.cat([batch_state[4] for batch_state in dec_state_list], dim=0),
                               None]
        # batch_dec_state = [torch.cat([batch_state[i] for batch_state in dec_state_list], dim=0) for i in range(6)]
        if dec_state_list[0][6][0] is None:
            batch_dec_state.append(dec_state_list[0][6])
        else:
            state_3 = []
            for i in range(len(dec_state_list[0][6])):
                state_3.append(torch.cat([batch_state[6][i] for batch_state in dec_state_list], dim=0))
            assert len(state_3) == len(dec_state_list[0][6])
            batch_dec_state.append(state_3)
        return batch_dec_state


class BeamSearchNode(object):
    def __init__(self, dec_state, previousNode, commentID, logProb, length, length_penalty=1):
        '''
        :param dec_state:
        :param previousNode:
        :param commentID:
        :param logProb:
        :param length:
        '''
        self.dec_state = dec_state
        self.prevNode = previousNode
        self.commentID = commentID
        self.logp = logProb
        self.leng = length
        self.length_penalty = length_penalty
        if self.prevNode is None:
            self.history_word = [int(commentID)]
            self.score = -100
        else:
            self.history_word = previousNode.history_word + [int(commentID)]
            self.score = self.eval()

    def eval(self):
        return self.logp / self.leng ** self.length_penalty
