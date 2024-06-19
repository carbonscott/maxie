import numpy as np
import psana

class PsanaImg:
    """
    It serves as an image accessing layer based on the data management system
    psana in LCLS.
    """

    def __init__(self, exp, run, mode, detector_name):

        # Biolerplate code to access an image
        # Set up data source
        self.datasource_id = f"exp={exp}:run={run}:{mode}"
        self.datasource    = psana.DataSource( self.datasource_id )
        self.run_current   = next(self.datasource.runs())
        self.timestamps    = self.run_current.times()

        # Set up detector
        self.detector = psana.Detector(detector_name)

        # Set image reading mode
        self.read = { "raw"   : self.detector.raw,
                      "calib" : self.detector.calib,
                      "image" : self.detector.image,
                      "mask"  : self.detector.mask, }

        # Set the flag about using a bad pixel mask
        self.cached_bad_pixel_mask = self.create_bad_pixel_mask()


    def __len__(self):
        return len(self.timestamps)


    def get(self, event_num, id_panel = None, mode = "calib"):
        # Fetch the timestamp according to event number...
        timestamp = self.timestamps[event_num]

        # Access each event based on timestamp...
        event = self.run_current.event(timestamp)

        # Only two modes are supported...
        assert mode in ("raw", "calib", "image"), \
            f"Mode {mode} is not allowed!!!  Only 'raw', 'calib' and 'image' are supported."

        # Fetch image data based on timestamp from detector...
        data = self.read[mode](event)
        img  = data[int(id_panel)] if id_panel is not None else data

        return img


    def assemble(self, multipanel = None, mode = "image", fake_event_num = 0):
        # Set up a fake event_num...
        event_num = fake_event_num

        # Fetch the timestamp according to event number...
        timestamp = self.timestamps[int(event_num)]

        # Access each event based on timestamp...
        event = self.run_current.event(timestamp)

        # Fetch image data based on timestamp from detector...
        img = self.read[mode](event, multipanel)

        return img


    def create_bad_pixel_mask(self):
        mask_bad_pixel = None
        try:
            mask_bad_pixel = self.read["mask"](
                self.run_current,
                calib       = True,
                status      = True,
                edges       = True,
                central     = True,
                unbond      = True,
                unbondnbrs  = True,
                unbondnbrs8 = False
            ).astype(np.uint16)
        except:
            print("Error in accessing the bad pixel mask!!!")

        return mask_bad_pixel


    def get_masked(self, event_num, id_panel = None, returns_assemble = False, edge_width = None):
        img = self.get(event_num, id_panel, 'calib')

        if edge_width is not None:
            img[..., :edge_width , :           ] = 0  # Top
            img[..., :           , :edge_width ] = 0  # Left
            img[..., -edge_width:, :           ] = 0  # Bottom
            img[..., :           , -edge_width:] = 0  # Right

        if self.cached_bad_pixel_mask is not None:
            img= PsanaImg.apply_mask(img, self.cached_bad_pixel_mask, 0.0)

        if returns_assemble:
            img = self.assemble(img)

        return img


    @staticmethod
    def apply_mask(data, mask, mask_value = np.nan):
        """
        Return masked data.

        Args:
            data: numpy.ndarray with the shape of (B, H, W).·
                  - B: batch of images.
                  - H: height of an image.
                  - W: width of an image.

            mask: numpy.ndarray with the shape of (B, H, W).·

        Returns:
            data_masked: numpy.ndarray.
        """
        # Mask unwanted pixels with np.nan...
        data_masked = np.where(mask, data, mask_value)

        return data_masked
