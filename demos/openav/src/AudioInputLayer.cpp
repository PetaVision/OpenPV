//Reads an audio stream from disk and prepares it for EarLayer
//Austin Thresher, 3/29/16
//Requires compilation with -std=c++11

#include <aquila/aquila.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <complex>
#include <algorithm>
#include <iostream>

#include <layers/HyPerLayer.hpp>
#include "AudioInputLayer.hpp"

AudioInputLayer::AudioInputLayer() 
{
  initialize_base();
}

AudioInputLayer::AudioInputLayer(char const * name, PV::HyPerCol * hc)
{
	initialize_base();
	initialize(name, hc);
}

AudioInputLayer::~AudioInputLayer() { }

int AudioInputLayer::initialize_base()
{
	samplerate = 44100;
	listening_time = 1000;
	window_stride = 64;
	current_wav.clear();
	generate_waveform = false;
	waveform_shape = WAVE_NONE;
	noise = 0.0f;
	playhead_randomness = 0.0f;
	shuffle_playlist = false;
	playlist_path = nullptr;
	playlist_index = 0;
	fft_layer = false;
	playback_layer = false;
	playback_time = 0;
	playhead_offset = 0.0f;
	playhead_position = 0;
	playhead_width = 0;
	playhead_advance = 1.0f;
	playhead_speed = 10;
	normalize_phase = false;
	return 0;
}

int AudioInputLayer::initialize(const char * name, PV::HyPerCol * hc)
{
	int status = PV::HyPerLayer::initialize(name, hc);
	return status;
}

int AudioInputLayer::allocateDataStructures()
{
	int status = PV::HyPerLayer::allocateDataStructures();
	current_wav.resize(parent->getNBatch());
	return status;
}

int AudioInputLayer::initializeActivity()
{
   int status = PV::HyPerLayer::initializeActivity();
   
   next_wav();
   update_buffer();
   
   return status;
}

void AudioInputLayer::initialize_playlist()
{
	if(generate_waveform) return; //We don't care about a playlist if we're synthesizing
	
	//Read in our playlist file one line at a time
	//and store it in our playlist vector.
	
	std::ifstream playlist_file(playlist_path);
	if(!playlist_file.is_open())
	{
		std::cerr << getName() << ": Could not open playlist: " << playlist_path << std::endl;
		exit(EXIT_FAILURE);
	}
	
	playlist.clear();
	shuffled_indices.clear();
	int index = 0;
	std::string path;
	
	while(std::getline(playlist_file, path))
	{
		playlist.push_back(path);
		shuffled_indices.push_back(index++);
	}
	if(playlist.size() <= 0)
	{
		std::cerr << getName() << ": Playlist file is empty: " << playlist_path << std::endl;
		exit(EXIT_FAILURE);
	}
	std::cout << getName() << ": Read " << playlist.size() << " playlist entries." << std::endl;
	
	if(shuffle_playlist)
	{
		std::random_shuffle(shuffled_indices.begin(), shuffled_indices.end());
	}
	
	playlist_index = 0;
}

int AudioInputLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag)
{
	ioParam_synth(ioFlag);
	ioParam_waveform(ioFlag);
	ioParam_noise(ioFlag);
	ioParam_playheadRandomness(ioFlag);
	ioParam_samplerate(ioFlag);
	ioParam_listeningTime(ioFlag);
	ioParam_play(ioFlag);
	ioParam_playheadOffset(ioFlag);
	ioParam_playheadAdvance(ioFlag);
	ioParam_playheadSpeed(ioFlag);
	ioParam_shuffle(ioFlag);
	ioParam_playlist(ioFlag);
	ioParam_fftLayer(ioFlag);
	ioParam_windowStride(ioFlag);
	
	return PV::HyPerLayer::ioParamsFillGroup(ioFlag);
}

double AudioInputLayer::getDeltaUpdateTime()
{
	if(!playback_layer) return listening_time;
	else return playhead_speed;
}

int AudioInputLayer::updateState(double time, double dt)
{
	if(fabs( time - (parent->getStartTime() + parent->getDeltaTime()) ) > parent->getDeltaTime()/2)
	{
		if(playback_layer)
		{
			playback_time += playhead_speed;
			float rand_factor = 1.0f;
			if(playhead_randomness > 0.0f) rand_factor = 1.0f - playhead_randomness/2.0f + Aquila::randomDouble() * playhead_randomness;
			playhead_position += static_cast<int>(playhead_advance * playhead_width * rand_factor);
		}
		if(!playback_layer || (playback_time >= listening_time))
		{
			std::cout << getName() << ": Switching wav at timestamp " << time << std::endl;
			next_wav(); //next_wav resets our playhead position and playback time
		}
		update_buffer();
	}
	return PV_SUCCESS;
}

void AudioInputLayer::update_buffer()
{
   #ifdef PV_USE_OPENMP_THREADS
		#pragma omp parallel for
	#endif
	for(int b = 0; b < parent->getNBatch(); b++)
	{
		if(!fft_layer && !playback_layer)
		{
			wav_to_buffer(b);
		}
		if(fft_layer && !playback_layer)
		{
			fft_to_buffer(b);
		}
		if(!fft_layer && playback_layer)
		{
			playback_wav_to_buffer(b);
		}
		if(fft_layer && playback_layer)
		{
			playback_fft_to_buffer(b);
		}
	}
}

void AudioInputLayer::next_wav()
{
   int nBatch = parent->getNBatch();

   #ifdef PV_USE_OPENMP_THREADS
      #pragma omp parallel for
   #endif
   for(int b = 0; b < nBatch; b++)
   {
      if(current_wav[b] != nullptr) //Free our old wav file before loading a new one
      {
         delete current_wav[b];
         current_wav[b] = nullptr;
      }
      int ind = -1;
      std::string fname;
      Aquila::SignalSource *tempwav;
      if(!generate_waveform) //Read a file from our playlist
      {
         //Different wav for each batch
         int seperation = playlist.size() / nBatch;
      
         ind = (b * seperation + playlist_index) % playlist.size();
         if(shuffle_playlist) ind = shuffled_indices[ind];
         
         fname = playlist[ind];
         tempwav = new Aquila::WaveFile(fname);
      }
      else //Generate a waveform
      {
         int max_wavelength = getLayerLoc()->nx;
         int buffer_size = max_wavelength;
         if(playback_layer) //TODO: Support FFT layer with generated waveforms
         {
            //Generate enough audio to listen for the entire listening time
            buffer_size = max_wavelength * (listening_time / playhead_speed * playhead_advance);
         }
         double start_phase = Aquila::randomDouble() * M_PI * 2.0;
         int wavelength = Aquila::random(max_wavelength / 32, max_wavelength);
         double delta = M_PI * 2.0 / wavelength;
         double *audio_data = new double[buffer_size];
         
         int attack_samples = Aquila::random(8, max_wavelength/2);
         int release_samples = Aquila::random(8, max_wavelength/2);
         
         Aquila::HannWindow attack(attack_samples);
         Aquila::HannWindow release(release_samples);
         int release_offset = buffer_size - release_samples / 2;
         
         #ifdef PV_USE_OPENMP_THREADS
            #pragma omp parallel for
         #endif   
         for(int s = 0; s < buffer_size; s++)
         {
            //Synthesize our waveform
            double phase = fmod(start_phase + s * delta, M_PI * 2.0);
            switch(waveform_shape)
            {
               case WAVE_SINE:
                  audio_data[s] = sin(phase);
                  break;
               case WAVE_SAW:
                  audio_data[s] = 1.0 - phase / M_PI;
                  break;
               case WAVE_TRIANGLE:
                  if(phase < M_PI) audio_data[s] = -1 + (2 / M_PI) * phase;
                  else audio_data[s] = 3 - (2 / M_PI) * phase;
                  break;
               case WAVE_SQUARE:
                  if(phase < M_PI) audio_data[s] = 1.0;
                  else audio_data[s] = -1.0;
                  break;
               default: audio_data[s] = 0.0f; //WAVE_NONE ends up here, in case we just want noise
            }
            
            //Apply envelopes
            if(s < attack_samples / 2) audio_data[s] *= attack.sample(s);
            if(s >= release_offset) audio_data[s] *= (1.0 - release.sample(s - release_offset));
            
            //Apply noise
            if(Aquila::randomDouble() <= noise) audio_data[s] += -noise/2.0f + Aquila::randomDouble() * noise/2.0f;
         }
         
         tempwav = new Aquila::SignalSource(audio_data, buffer_size, samplerate);
         fname = "Synthesized Waveform";
      }
      
      
      //Now let's normalize our audio
      double max = 0;
      for(int i = 0; i < tempwav->getSamplesCount(); i++)
      {
         double s = abs(tempwav->sample(i));
         max = s > max ? s : max;
      }
      if(max != 0) { (*tempwav) *= 1.0f / max; }
      
      //Assign this audio to its batch
      current_wav[b] = tempwav;
		pvInfo() << getName() << ": Loaded " << fname << " into batch " << b << " (playlist entry " << ind << ")" << std::endl;
	}
   
   if(!generate_waveform)
   {
   	playlist_index++;
		if(playlist_index >= playlist.size())
		{
			std::cout << getName() << ": Playlist has reached its end, looping." << playlist_path << std::endl;
			playlist_index = 0;
		}
   }
	
	if(playback_layer)
	{
		if(fft_layer)
		{
			playhead_width = getLayerLoc()->nx * window_stride;
		}
		else
		{
			playhead_width = getLayerLoc()->nx;
		}
		
		playhead_position = static_cast<int>(playhead_offset * playhead_width);
		playback_time = 0;
		std::cout << getName() << ": Placing playhead at " << playhead_position << ", playhead size is " << playhead_width << std::endl;
	}
}

void AudioInputLayer::wav_to_buffer(int batchIdx)
{
	const PVLayerLoc * loc = getLayerLoc();
	const int nx = loc->nx, ny = loc->ny, nf = loc->nf;
	const PVHalo * halo = &loc->halo;
	pvdata_t * activityBuffer = clayer->activity->data;

	#ifdef PV_USE_OPENMP_THREADS
		#pragma omp parallel for
	#endif
	for(int n = 0; n < nx; n++)
	{
		activityBuffer[kIndexBatch(batchIdx, halo->lt+n, halo->up, 0, parent->getNBatch(), nx + halo->lt + halo->rt, ny + halo->dn + halo->up, nf)]
			= read_resampled(current_wav[batchIdx], n);
	}
}

void AudioInputLayer::playback_wav_to_buffer(int batchIdx)
{
	const PVLayerLoc * loc = getLayerLoc();
	const int nx = loc->nx, ny = loc->ny, nf = loc->nf;
	const PVHalo * halo = &loc->halo;
	pvdata_t * activityBuffer = clayer->activity->data;

	#ifdef PV_USE_OPENMP_THREADS
		#pragma omp parallel for
	#endif
	for(int n = 0; n < nx; n++)
	{
		activityBuffer[kIndexBatch(batchIdx, halo->lt+n, halo->up, 0, parent->getNBatch(), nx + halo->lt + halo->rt, ny + halo->dn + halo->up, nf)]
			= read_resampled(current_wav[batchIdx], playhead_position+n);
	}
}

void AudioInputLayer::fft_to_buffer(int batchIdx)
{
	const PVLayerLoc * loc = getLayerLoc();
	const int nx = loc->nx, ny = loc->ny, nf = loc->nf;
	const PVHalo * halo = &loc->halo;
	pvdata_t * activityBuffer = clayer->activity->data;

	int window_size = Aquila::nextPowerOf2(ny);
   
   if(ny < window_size / 2)
   {
      pvError() << "Invalid ny (" << ny << ", smaller than FFT size " << window_size / 2 << "). Must be a power of 2." << std::endl;
      throw;
   }
   
	#ifdef PV_USE_OPENMP_THREADS
		#pragma omp parallel for
	#endif
	for(int w = 0; w < nx; w++)
	{
		Aquila::SpectrumType spectrum = fftChunk(current_wav[batchIdx], w * window_stride, window_size);
      float sum = 0;
		for(int i = 0; i < window_size / 2; i++)
		{
         int ind = kIndexBatch(batchIdx, halo->lt+w, halo->up+i, 0, parent->getNBatch(), nx + halo->lt + halo->rt, ny + halo->dn + halo->up, nf);
         float hyp = abs(spectrum[i]);
			switch(nf)
         {
            case 1: //Just power spectrum
               activityBuffer[ind] = hyp / window_size;
               break;
            case 2: //Complex values
               activityBuffer[ind] = spectrum[i].real();
               activityBuffer[ind + 1] = spectrum[i].imag();
               break;
            case 3: //Phase as a (potentially normalized) vector and power as a scalar
               if(!normalize_phase)
               {
                  activityBuffer[ind] = spectrum[i].real();
                  activityBuffer[ind + 1] = spectrum[i].imag();
               }
               else
               {
                  activityBuffer[ind] = spectrum[i].real() / hyp;
                  activityBuffer[ind + 1] = spectrum[i].imag() / hyp;
               }
               activityBuffer[ind + 2] = hyp / window_size;
               break;
         }
		}
	}
}

void AudioInputLayer::playback_fft_to_buffer(int batchIdx)
{
	const PVLayerLoc * loc = getLayerLoc();
	const int nx = loc->nx, ny = loc->ny, nf = loc->nf;
	const PVHalo * halo = &loc->halo;
	pvdata_t * activityBuffer = clayer->activity->data;

	int window_size = Aquila::nextPowerOf2(ny);

   if(ny < window_size / 2)
   {
      pvError() << "Invalid ny (" << ny << ", smaller than FFT size " << window_size / 2 << "). Must be a power of 2." << std::endl;
      throw;
   }

	#ifdef PV_USE_OPENMP_THREADS
		#pragma omp parallel for
	#endif
	for(int w = 0; w < nx; w++)
	{
		Aquila::SpectrumType spectrum = fftChunk(current_wav[batchIdx], playhead_position + w * window_stride, window_size);
      float sum = 0;
		for(int i = 0; i < window_size / 2; i++)
		{
         int ind = kIndexBatch(batchIdx, halo->lt+w, halo->up+i, 0, parent->getNBatch(), nx + halo->lt + halo->rt, ny + halo->dn + halo->up, nf);
         float hyp = abs(spectrum[i]);
			switch(nf)
         {
            case 1: //Just power spectrum
               activityBuffer[ind] = hyp / window_size;
               break;
            case 2: //Complex values
               activityBuffer[ind] = spectrum[i].real();
               activityBuffer[ind + 1] = spectrum[i].imag();
               break;
            case 3: //Phase as a (potentially normalized) vector and power as a scalar
               if(!normalize_phase)
               {
                  activityBuffer[ind] = spectrum[i].real();
                  activityBuffer[ind + 1] = spectrum[i].imag();
               }
               else
               {
                  activityBuffer[ind] = spectrum[i].real() / hyp;
                  activityBuffer[ind + 1] = spectrum[i].imag() / hyp;
               }
               activityBuffer[ind + 2] = hyp / window_size;
               break;
         }
		}
	}
}

Aquila::SpectrumType AudioInputLayer::fftChunk(Aquila::SignalSource* data, int start_sample, int window_size)
{
	Aquila::HammingWindow window(window_size);
	auto fft = Aquila::FftFactory::getFft(window_size);
	double audio[window_size];
	for(int n = 0; n < window_size; n++) { audio[n] = read_resampled(data, start_sample + n) * window.sample(n); }
	return fft->fft(audio);
}

float AudioInputLayer::read_resampled(Aquila::SignalSource* wav, int n)
{
	float sample_step = wav->getSampleFrequency() * 1.0f / samplerate;
	if(sample_step == 1.0)
	{
		if(n >= wav->getSamplesCount()) return 0;
		return wav->sample(n);
	}
	
	int endIndex = static_cast<int>((n+1) * sample_step);
	int beginIndex = static_cast<int>(n * sample_step);
	if(beginIndex < 0) beginIndex = 0;
	if(endIndex >= wav->getSamplesCount()) endIndex = wav->getSamplesCount()-1;
	if(endIndex == beginIndex) return wav->sample(endIndex);
	if(beginIndex > endIndex) return 0;
	
	float remainder = (n * sample_step) - beginIndex;
	
	double start = wav->sample(beginIndex);
	double end = wav->sample(endIndex);
	return static_cast<float>(start + (end - start) * remainder);
}

//Params
void AudioInputLayer::ioParam_samplerate(enum ParamsIOFlag ioFlag)
	{ parent->ioParamValue(ioFlag, name, "samplerate", &samplerate, samplerate); }
void AudioInputLayer::ioParam_listeningTime(enum ParamsIOFlag ioFlag)
	{ parent->ioParamValue(ioFlag, name, "listeningTime", &listening_time, listening_time); }
void AudioInputLayer::ioParam_windowStride(enum ParamsIOFlag ioFlag)
	{ parent->ioParamValue(ioFlag, name, "windowStride", &window_stride, window_stride); }
void AudioInputLayer::ioParam_fftLayer(enum ParamsIOFlag ioFlag)
	{ parent->ioParamValue(ioFlag, name, "fftLayer", &fft_layer, fft_layer); }
void AudioInputLayer::ioParam_shuffle(enum ParamsIOFlag ioFlag)
	{ parent->ioParamValue(ioFlag, name, "shuffle", &shuffle_playlist, shuffle_playlist); }
void AudioInputLayer::ioParam_play(enum ParamsIOFlag ioFlag)
	{ parent->ioParamValue(ioFlag, name, "play", &playback_layer, playback_layer); }
void AudioInputLayer::ioParam_synth(enum ParamsIOFlag ioFlag)
	{ parent->ioParamValue(ioFlag, name, "synth", &generate_waveform, generate_waveform); }
void AudioInputLayer::ioParam_playheadOffset(enum ParamsIOFlag ioFlag)
	{ parent->ioParamValue(ioFlag, name, "playheadOffset", &playhead_offset, playhead_offset); }
void AudioInputLayer::ioParam_playheadAdvance(enum ParamsIOFlag ioFlag)
	{ parent->ioParamValue(ioFlag, name, "playheadAdvance", &playhead_advance, playhead_advance); }
void AudioInputLayer::ioParam_playheadSpeed(enum ParamsIOFlag ioFlag)
	{ parent->ioParamValue(ioFlag, name, "playheadSpeed", &playhead_speed, playhead_speed); }
void AudioInputLayer::ioParam_noise(enum ParamsIOFlag ioFlag)
	{ parent->ioParamValue(ioFlag, name, "noise", &noise, noise); }
void AudioInputLayer::ioParam_playheadRandomness(enum ParamsIOFlag ioFlag)
	{ parent->ioParamValue(ioFlag, name, "playheadRandomness", &playhead_randomness, playhead_randomness); }
void AudioInputLayer::ioParam_normalizePhase(enum ParamsIOFlag ioFlag)
	{ parent->ioParamValue(ioFlag, name, "normalizePhase", &normalize_phase, normalize_phase); }
void AudioInputLayer::ioParam_waveform(enum ParamsIOFlag ioFlag)
{
	if (ioFlag == PARAMS_IO_READ)
	{
		char *waveshape;
		parent->ioParamString(ioFlag, name, "waveform", &waveshape, "none", true);
		if(!strcmp(waveshape, "sine")) waveform_shape = WAVE_SINE;
		else if(!strcmp(waveshape, "saw")) waveform_shape = WAVE_SAW;
		else if(!strcmp(waveshape, "square")) waveform_shape = WAVE_SQUARE;
		else if(!strcmp(waveshape, "triangle")) waveform_shape = WAVE_TRIANGLE;
		else waveform_shape = WAVE_NONE;
	}
}
void AudioInputLayer::ioParam_playlist(enum ParamsIOFlag ioFlag)
{
	if (ioFlag == PARAMS_IO_READ)
	{
		parent->ioParamStringRequired(ioFlag, name, "playlist", &playlist_path);
		initialize_playlist();
	}
}
