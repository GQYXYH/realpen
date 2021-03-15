import time
import numpy as np
import cv2

try:
    import PySpin
except ImportError:
    print("Warning: Package 'PySpin' not found. Please install it via the Spinnaker SDK from Flir.")


"""
Class for using FLIR Blackfly camera

The camera buffer is set to NewestOnly which allows real-time image acquisition without delay. Furthermore the exposure time
can be set but the default is set to maximize the frame rate to approx. 500 fps. Before grabbing images acquisition needs to 
be started. To avoid errors the camera should be used like it is done in run_camera()
"""
######################################################################
# SET PARAMETERS
exposure_time = 1000 # less than 1000 does not reduce the exposure time anymore
######################################################################


class Blackfly:

    def __init__(self, exposure_time=exposure_time):
        # Retrieve singleton reference to system object
        self.system = PySpin.System.GetInstance()

        # Retrieve list of cameras from the system
        self.cam_list = self.system.GetCameras()

        num_cameras = self.cam_list.GetSize()

        # Finish if there are no cameras
        if num_cameras == 0:
            # Clear camera list before releasing system
            self.cam_list.Clear()
            # Release system instance
            self.system.ReleaseInstance()
            print('Not enough cameras!')

        self.cam = self.cam_list[0]
        self.cam.Init()

        # Set exposure time
        self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        self.cam.ExposureTime.SetValue(exposure_time)
        # self.cam.ExposureTime.SetValue(self.cam.ExposureTime.GetMin())
        self.cam.GainAuto.SetValue(PySpin.GainAuto_Continuous)
        self.cam.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Continuous)
        self.cam.BalanceWhiteAutoProfile.SetValue(PySpin.BalanceWhiteAutoProfile_Indoor)
        self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BayerRG8)
        self.cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit8)
        self.cam.BinningSelector.SetValue(PySpin.BinningSelector_All)
        # self.cam.ColorTransformationSelector.SetValue(PySpin.ColorTransformationSelector_RGBtoRGB)

        # Set Buffer to newest first
        handling_mode = PySpin.CEnumerationPtr(self.cam.GetTLStreamNodeMap().GetNode('StreamBufferHandlingMode'))
        handling_mode_entry = handling_mode.GetEntryByName('NewestOnly')
        handling_mode.SetIntValue(handling_mode_entry.GetValue())

        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

        # Set stream buffer Count Mode to auto
        s_node_map = self.cam.GetTLStreamNodeMap()
        stream_buffer_count_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferCountMode'))
        stream_buffer_count_mode.SetIntValue(
            PySpin.CEnumEntryPtr(stream_buffer_count_mode.GetEntryByName('Auto')).GetValue())


        # Change width, height, OffsetY, OffsetX to just get part of capture
        # self.cam.Width.SetValue(220)
        # print(self.cam.Width.GetValue())


        # example for how to use the pointers
        # nodemap = self.cam.GetNodeMap()
        # exposureAuto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
        # exposureAuto.SetIntValue(exposureAuto.GetEntryByName("Off").GetValue())
        self.time_delays = []

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time_delays = np.array(self.time_delays)
        mean = np.mean(self.time_delays) * 1000
        std = np.std(self.time_delays) * 1000
        print("Capture time of Blackfly (ms): mean = " + str(mean) + ", std = " + str(std))
        try:
            self.cam.DeInit()
        except:
            self.end_acquisition()
            self.cam.DeInit()
            print("Did not end acquisition")

        del self.cam

        # Clear camera list before releasing system
        self.cam_list.Clear()

        del self.cam_list

        # Release system instance
        self.system.ReleaseInstance()

    def __enter__(self):
        return self

    def start_acquisition(self):
        # nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
        # nodemap = self.cam.GetNodeMap()
        # node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        # node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        # acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
        # node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
        self.cam.BeginAcquisition()
        return

    def end_acquisition(self):
        try:
            self.cam.EndAcquisition()
        except:
            print('endAcquisition failed')

        return

    def get_image(self):
        try:
            t = time.time()
            image_result = self.cam.GetNextImage()
            if image_result.IsIncomplete():
                print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                return
            else:
                # width = image_result.GetWidth()
                # height = image_result.GetHeight()
                # print('Grabbed Image, width = %d, height = %d' % (width, height))
                rawFrame = np.array(image_result.GetData(), dtype="uint8").reshape(
                    (image_result.GetHeight(), image_result.GetWidth()))
                frame = cv2.cvtColor(rawFrame, cv2.COLOR_BAYER_BG2BGR)
                # image_result.Save('image0.png')
                image_result.Release()
            self.time_delays.append(time.time() - t)

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return None

        return frame


def reset_blackfly():
    with Blackfly() as bf:
        try:
            bf.start_acquisition()
            bf.get_image()
        except:
            pass
        try:
            bf.end_acquisition()
        except:
            pass
    print("Camera reseted")
    return


def run_camera():
    with Blackfly() as bf:
        print("Camera is running, press 'q' to exit.")
        bf.start_acquisition()

        # frames can be captured now
        while (True):
            # Capture frame-by-frame
            frame = bf.get_image()
            cv2.imshow('frame', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

        bf.end_acquisition()

def get_one_frame():
    with Blackfly() as bf:
        print("Camera is running, press 'q' to exit.")
        bf.start_acquisition()
        frame = bf.get_image()
        filepath = "example_image.png"
        cv2.imwrite(filepath, frame)
        bf.end_acquisition()
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_camera()
    # get_one_frame()
