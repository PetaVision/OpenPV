#pragma once

#include <string>
#include <vector>
#include <aquila/aquila.h>
#include <layers/HyPerLayer.hpp>

class AudioInputLayer : public PV::HyPerLayer
{

public:
	AudioInputLayer(char const * name, PV::HyPerCol * hc);
	virtual ~AudioInputLayer();
	virtual bool activityIsSpiking() { return false; }
	virtual int allocateDataStructures();
	virtual double getDeltaUpdateTime();
	virtual int updateState(double time, double dt);

protected:
	AudioInputLayer();
	int initialize(char const * name, PV::HyPerCol * hc);
	virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
	virtual int initializeActivity();

	enum WaveShape
	{
		WAVE_NONE, WAVE_SINE, WAVE_SAW, WAVE_SQUARE, WAVE_TRIANGLE
	};
	
	float read_resampled(Aquila::SignalSource* wav, int n);
	Aquila::SpectrumType fftChunk(Aquila::SignalSource* data, int start_sample, int window_size);
	
	int samplerate;
	int window_stride;
	int listening_time;
	bool fft_layer;
	int playlist_index;
	bool shuffle_playlist;
	bool generate_waveform;
	WaveShape waveform_shape;
	float noise;
	char* playlist_path;
	std::vector<std::string> playlist;
	std::vector<int> shuffled_indices;
	std::vector<Aquila::SignalSource*> current_wav;
	bool playback_layer;
   bool normalize_phase;
	float playhead_offset;
	int playhead_position; //Where is the starting sample for our playhead
	int playhead_width;
	float playhead_advance; //How many widths to advance the playhead
	float playhead_randomness; //How much randomness is applied to the playhead advance
	int playhead_speed; //How many timesteps to wait between advances
	int playback_time; //Used to keep track of the number of timesteps between playlist changes when using playback mode
	
	
private:
	int initialize_base();
	void initialize_playlist();
	void next_wav();
	void update_buffer();
	void wav_to_buffer(int batchIdx);
	void fft_to_buffer(int batchIdx);
	void playback_wav_to_buffer(int batchIdx);
	void playback_fft_to_buffer(int batchIdx);
	
	
/**
* List of parameters needed from the Movie class
* @name AudioInput Parameters
* @{
*/
public:
	/**
	* @brief samplerate: What samplerate to convert to when reading in data.
	*/
	virtual void ioParam_samplerate(enum ParamsIOFlag ioFlag);
	/**
	* @brief listeningTime: Number of timesteps to listen to each playlist entry
	*/
	virtual void ioParam_listeningTime(enum ParamsIOFlag ioFlag);
	/**
	* @brief playlist: Path to a txt file with a list of wav files to train on
	*/
	virtual void ioParam_playlist(enum ParamsIOFlag ioFlag);
	/**
	* @brief shuffle: If true, randomize the order of the playlist. I hope this uses the random seed from the Column.
	*/
	virtual void ioParam_shuffle(enum ParamsIOFlag ioFlag);
	/**
	* @brief fftLayer: Take the FFT of the input instead of passing raw samples.
	* 	Requires nf = 2.
	* 	nf(1) is Amplitude, nf(2) is Phase
	* 	Window size is the power of 2 smaller than or equal to ny
	*/
	virtual void ioParam_fftLayer(enum ParamsIOFlag ioFlag);
	/**
	* @brief windowStride: If this is an FFT layer, how many samples to advance 
	*/
	virtual void ioParam_windowStride(enum ParamsIOFlag ioFlag);
	/**
	* @brief play: Play the entire audio file instead of just the first X samples.
	* listeningTime is how long to listen to one chunk before advancing the playhead.
	* playhead length is based on nx.
	*/
	virtual void ioParam_play(enum ParamsIOFlag ioFlag);
	/**
	* @brief playheadOffset: How many playhead lengths is this offset by
	* on the audio stream? +1.0 is one full playhead length in front, -0.5
	* is half of a playhead length behind, etc. 
	*/
	virtual void ioParam_playheadOffset(enum ParamsIOFlag ioFlag);
	/**
	* @brief playheadAdvance: How many playhead lengths to advance each playheadSpeed.
	* Works the same as playheadOffset
	*/
	virtual void ioParam_playheadAdvance(enum ParamsIOFlag ioFlag);
	/**
	* @brief playheadSpeed: How many timesteps pass before advancing the playhead
	*/
	virtual void ioParam_playheadSpeed(enum ParamsIOFlag ioFlag);
	/**
	 * @brief synthesize: Instead of reading wavs from a playlist, generate
	 * random waves to train on. Wavelength, phase, and envelope are randomized.
	 */
	virtual void ioParam_synth(enum ParamsIOFlag ioFlag); 
	/**
	* @brief waveform: What waveform shape to synthesize, if synthesize = true.
	* Accepted strings are "sine", "square", "triangle", "saw", and "none".
	*/
	virtual void ioParam_waveform(enum ParamsIOFlag ioFlag);
	/**
	* @brief noise: How much white noise to add to a sine wav
	*/
	virtual void ioParam_noise(enum ParamsIOFlag ioFlag);
	/**
	* @brief playheadRandomness: Scales playhead advances by +- random%
	*/
	virtual void ioParam_playheadRandomness(enum ParamsIOFlag ioFlag);
   virtual void ioParam_normalizePhase(enum ParamsIOFlag ioFlag); 
/** @} */
};
