import openvino as ov

def query_device():
    """
    Each device has several properties as seen in the last command. Some of the key properties are:
    - `FULL_DEVICE_NAME` - The product name of the GPU and whether it is an integrated or discrete GPU (iGPU or dGPU).
    - `OPTIMIZATION_CAPABILITIES` - The model data types (INT8, FP16, FP32, etc) that are supported by this GPU.
    - `GPU_EXECUTION_UNITS_COUNT` - The execution cores available in the GPU's architecture, which is a relative measure of the GPU's processing power.
    - `RANGE_FOR_STREAMS` - The number of processing streams available on the GPU that can be used to execute parallel inference requests. When compiling a model in LATENCY or THROUGHPUT mode, OpenVINO will automatically select the best number of streams for low latency or high throughput.
    - `PERFORMANCE_HINT` - A high-level way to tune the device for a specific performance metric, such as latency or throughput, without worrying about device-specific settings.
    - `CACHE_DIR` - The directory where the model cache data is stored to speed up compilation time.

    To learn more about devices and properties, see the [Query Device Properties](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties.html) page.
    """
    core = ov.Core()
    available_devices = core.available_devices
    print(" -- available devices: ", available_devices)

    for device in available_devices:
        print("=" * 32)
        print(f' -- Device "{device}" SUPPORTED_PROPERTIES:')
        supported_properties = core.get_property(device, "SUPPORTED_PROPERTIES")
        indent = len(max(supported_properties, key=len))
        for property_key in supported_properties:
            if property_key not in ("SUPPORTED_METRICS", "SUPPORTED_CONFIG_KEYS", "SUPPORTED_PROPERTIES"):
                try:
                    property_val = core.get_property(device, property_key)
                except TypeError:
                    property_val = "UNSUPPORTED TYPE"
                print(f"{property_key:<{indent}}: {property_val}")

if __name__ == "__main__":
    query_device()
