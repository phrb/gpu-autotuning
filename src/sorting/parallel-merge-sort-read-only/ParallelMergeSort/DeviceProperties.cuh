#ifndef DEVICE_PROPERTIES
#define DEVICE_PROPERTIES

class DeviceProperties
{
public:
	DeviceProperties();
	size_t SharedMemoryPerBlock;
	int MaxNumberOfBlocks;
	int MaxThreadsPerBlock;
	int ClockFrequency; // khz
	static void CheckCompatability();
};

#endif