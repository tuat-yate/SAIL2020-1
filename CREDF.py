# -*- coding: utf-8 -*-
#
# CREDF.py
#
# Created by Kosuke FUKUMORI on 2018/03/02
#

import datetime
import copy
import mne
import itertools
import numpy as np
import scipy.signal

### CONSTANTS ###
KEY_SAMP_INT =  'samp_int'
KEY_SAMP_FREQ = 'samp_freq'
KEY_MEAS_DATE = 'meas_date'
KEY_CH_NAMES =  'ch_names'
KEY_EDF_NAME =  'edf_name'
ONE_EPOCH_TIME = 20 # 20[s]

class _CREDF(object):
    @property
    def samp_int(self): return self._info[KEY_SAMP_INT]

    @property
    def samp_freq(self): return self._info[KEY_SAMP_FREQ]

    @property
    def meas_date(self):
        return self._info[KEY_MEAS_DATE].strftime('%Y/%m/%d %H:%M:%S')

    @property
    def ch_names(self): return self._info[KEY_CH_NAMES]

    @property
    def n_channels(self): return self._raw.shape[0]

    @property
    def n_bands(self): return self._raw.shape[1]

    @property
    def n_samples(self): return self._raw.shape[2]

    @property
    def edf_name(self): return self._info[KEY_EDF_NAME]

    @property
    def raw(self): return self._raw
    @raw.setter
    def raw(self, value):
        if self._raw is None: self._raw = value

    def __init__(self, edf_path=None, active_matches=None, credf=None, include_raw=True):
        '''
            CREDFは以下の初期化方法が用意されています．
            [EDFファイルから初期化する場合]
                EDFファイルを指定して初期化します．
                Required parameter(s)
                - edf_path: str -> edfファイルのパスを指定
                Optional parameter(s)
                - active_matches: None or [str] ->
                    None の場合:
                        全チャンネルを読み込みます
                    [str] の場合:
                        指定した文字列にマッチするチャンネルのみ読み込みます．
                        （リストに含まれる条件の"いずれか"に適合する場合）
                        文字列フォーマットは次の通りです:
                        "HOGE*" -> 前方一致
                        "*HOGE*" -> 部分一致
                        "*HOGE" -> 後方一致
                        "HOGE" -> 完全一致
            [他のCREDFから初期化する場合]
                CREDFインスタンスを指定して初期化します．
                rawデータをコピーしない場合は慎重な取扱いが必要になります．
                Required parameter(s)
                - credf: CREDF -> 他のCREDFインスタンス
                Optional parameter(s)
                - include_raw: bool ->
                    True の場合: rawデータを含めて完全にコピーされます
                    False の場合: rawデータはコピーされません
        '''
        if edf_path is None and credf is None:
            raise SyntaxError('Neither edf_path nor credf is undefined')
        elif edf_path is not None and credf is not None:
            raise SyntaxError('Both edf_path and credf are defined')
        elif edf_path is not None:
            self._init_from_edf(edf_path, active_matches)
        else:
            self._init_from_credf(credf, include_raw)

    def _init_from_edf(self, edf_path, active_matches):
        # MNEを利用してEDFを読み込み
        # (preloadは初期値Falseのためrawデータはメモリ上に展開されません)
        edf = mne.io.read_raw_edf(input_fname=edf_path, stim_channel=None, verbose='ERROR')

        # 利用しないチャンネル名一覧を取得
        if active_matches is not None:
            def is_ch_name_active(ch_name, conditions, rule):
                is_active = False
                for condition in conditions:
                    is_active |= rule(condition, ch_name)
                    if is_active: break
                return is_active
            excluded_channels = [
                name for name in edf.ch_names\
                if not is_ch_name_active(name, active_matches, lambda cond, ch_name:\
                    cond[-1] == '*' and cond[0] != '*' and ch_name.startswith(cond[:-1]) or \
                    cond[-1] != '*' and cond[0] == '*' and ch_name.endswith(cond[1:]) or \
                    cond[-1] == '*' and cond[0] == '*' and (cond in ch_name) or \
                    cond[-1] != '*' and cond[0] != '*' and (cond == ch_name))
            ]

        # MNEを利用してEDFを再読み込み
        # (ここでは除外チャンネルを読み込みません)
        edf.close()
        edf = mne.io.read_raw_edf(
            input_fname=edf_path,
            stim_channel=None,
            exclude=[] if active_matches is None else excluded_channels,
            verbose='ERROR')

        # サンプリング周波数と周期を取得
        if edf.info['sfreq'] is not None:
            sampling_frequency = edf.info['sfreq']
            sampling_interval = 1.0 / sampling_frequency
        else:
            sampling_interval = edf[0][1][1] - edf[0][1][0]
            sampling_frequency = 1.0 / sampling_interval

        # 測定日時を取得 (mne 0.17.0 以上ではタプルで返されるので要素分解)
        try: measured_date = edf.info['meas_date']
        except: measured_date = 0
        else:
            if mne.__version__ >= '0.17.0': measured_date, _ = measured_date
        measured_date_dt = datetime.datetime.fromtimestamp(measured_date)

        # EDFの情報を整理
        self._info = {
            KEY_SAMP_INT:  sampling_interval,
            KEY_SAMP_FREQ: int(sampling_frequency),
            KEY_MEAS_DATE: measured_date_dt,
            KEY_CH_NAMES:  edf.ch_names,
            KEY_EDF_NAME:  edf_path.split('/')[-1]
        }

        # np.array形式でrawデータを読み込み
        # rawのshapeは[n_channels, n_bands, n_samples]です
        self._raw = edf.get_data()[:, None, :]

    def _init_from_credf(self, credf, include_raw):
        self._info = credf._info.copy()
        self._raw = credf._raw.copy() if include_raw else None

    def get_ch_names_from(self, indices):
        if isinstance(indices, list):
            return [self.ch_names[index] for index in indices]
        else:
            return [self.ch_names[indices]]

    def get_indices_from(self, names):
        if isinstance(names, list):
            indices = [self.ch_names.index(name) for name in names]
        else:
            indices = [self.ch_names.index(names)]
        return indices

    def get_raw_from(self, names):
        return self._raw[self.get_indices_from(names)]

    def split_into(self, N, ch_indices=None):
        '''
            rawデータをNサンプルごとに分割します．
            ---------------------
            N: int -> 1エポックごとのサンプル数
            ch_indices: [str] or None -> 使用するチャンネルのindexを指定, Noneの場合は全チャンネル
            ---------------------
            出力は
            [n_channels, エポック数, N]
            の形式をとります
        '''
        N = int(N)
        n_epochs = self.n_samples // N
        solved_ch_indices = list(range(self.n_channels)) if ch_indices is None else ch_indices
        truncated_data = self._raw[solved_ch_indices, :, 0:n_epochs*N]
        return truncated_data.reshape(-1, self.n_bands, n_epochs, N)

    def split_into_20secs(self, ch_indices=None):
        '''
            rawデータを20秒ごとに分割します．
            ---------------------
            ch_indices: [str] or None -> 使用するチャンネルのindexを指定, Noneの場合は全チャンネル
            ---------------------
            出力は
            [n_channels, エポック数, self.samp_freq * ONE_EPOCH_TIME]
            の形式をとります
        '''
        return self.split_into(self.samp_freq*ONE_EPOCH_TIME, ch_indices)

    def generate_Hankel_mtrx(self, look_back, epoch_start, epoch_stop, epoch_step=1, ch_indices=None, dtype='float32'):
        '''
            lookbackを有効にしてエポック分割をします．
            各チャンネルごとに[エポック数, lookback数]からなる行列(=ハンケル行列)を作成します．
            エポック数は(epoch_stop-epoch_start)//epoch_stepです．
            ---------------------
            look_back: int -> 1エポックのサンプル数
            epoch_start,epoch_stop,epoch_step: int,int,int -> エポックの開始,終了,ステップ数
            ch_indices: [str] or None -> 使用するチャンネルのindexを指定, Noneの場合は全チャンネル
            ---------------------
            出力は
            [チャンネル数][エポック数][lookback]
            の形式をとります．1チャンネルあたり以下のような構成になります．
            [[t0, t1, t2],
             [t1, t2, t3],
             [t2, t3, t4],
             [t3, t4, t5]]
        '''
        start = max(epoch_start, 0)
        stop = min(epoch_stop, self.n_samples - look_back + 1)
        step = max(epoch_step, 1)
        n_epochs = (stop-start) // step
        if (stop-start) % step > 0: n_epochs += 1
        if ch_indices is None:
            ch_idx_mod = self.get_indices_from(self.ch_names)
        else:
            ch_idx_mod = ch_indices
        n_channels = len(ch_idx_mod)
        X = np.empty((n_channels, self.n_bands, n_epochs, look_back), dtype=dtype)
        for i, epoch in enumerate(range(start, stop, step)):
            X[:, :, i] = self._raw[ch_idx_mod, :, epoch:epoch+look_back]
        return X

    def apply_bandpass(self, lowcut, highcut, order):
        applied_bandpass = self.__class__(credf=self, include_raw=False)
        nyq = 0.5 * self.samp_freq
        low = lowcut / nyq
        high = highcut / nyq
        b, a = scipy.signal.butter(order, [low, high], btype='bandpass')
        applied_bandpass.raw = np.empty_like(self._raw)
        for ch, band in itertools.product(range(self.n_channels), range(self.n_bands)):
            applied_bandpass._raw[ch, band] = scipy.signal.filtfilt(b, a, self._raw[ch, band])
        return applied_bandpass

    def apply_lowpass(self, cutoff, order):
        applied_lowpass = self.__class__(credf=self, include_raw=False)
        nyq = 0.5 * self.samp_freq
        normal_cutoff = cutoff / nyq
        b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
        applied_lowpass.raw = np.empty_like(self._raw)
        for ch, band in itertools.product(range(self.n_channels), range(self.n_bands)):
            applied_lowpass._raw[ch, band] = scipy.signal.lfilter(b, a, self._raw[ch, band])
        return applied_lowpass

    def apply_notch(self, freq):
        applied_notch = self.__class__(credf=self, include_raw=False)
        applied_notch.raw = np.empty_like(self._raw)
        for ch, band in itertools.product(range(self.n_channels), range(self.n_bands)):
            applied_notch._raw[ch, band] = \
                mne.filter.notch_filter(self._raw[ch, band],
                                        Fs=self.samp_freq,
                                        freqs=freq,
                                        verbose=False,
                                        method='iir')
        return applied_notch

    def clip_samples(self, start, end):
        clipped_edf = self.__class__(credf=self, include_raw=False)
        clipped_edf.raw = np.array(self._raw[:, :, start:end])
        return clipped_edf

    def thin_out_samples(self, step):
        thinned_edf = self.__class__(credf=self, include_raw=False)
        thinned_edf._info[KEY_SAMP_INT] = thinned_edf._info[KEY_SAMP_INT] * step
        thinned_edf._info[KEY_SAMP_FREQ] = thinned_edf._info[KEY_SAMP_FREQ] // step
        thinned_edf.raw = np.array(self._raw[:, :, ::step])
        return thinned_edf

    def __repr__(self):
        return 'CREDF: {} | n_channels: {} | n_bands: {} | n_samples: {}'.format(\
            self.edf_name, self.n_channels, self.n_bands, self.n_samples)

    '''
        numpy-likeに使える様にする
    '''
    @property
    def shape(self): return self._raw.shape
    @property
    def ndim(self): return self._raw.ndim
    def __array__(self): return self._raw
    def __len__(self): return self._raw.__len__()
    def __getitem__(self, key): return self.raw.__getitem__(key)
