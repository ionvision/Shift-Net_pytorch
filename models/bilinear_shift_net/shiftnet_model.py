from models.shift_net.shiftnet_model import ShiftNetModel


class BilinearShiftNet(ShiftNetModel):
    def name(self):
        return 'BilinearShiftNet'