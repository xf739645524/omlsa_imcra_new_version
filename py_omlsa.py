import numpy as np
import scipy
from scipy.io import wavfile
import sys
import scipy.integrate as si
import scipy.special as sc
import librosa
import matplotlib.pyplot as plt
#np.set_printoptions(precision=3)
class imcra:
    def __init__(self, fs):
        # 1  fft param
        self.Fs_ref = 16000
        self.M_ref = 512
        self.Mo_ref = int(self.M_ref * 0.75)

        # 2  noise spectrum Estimation Param
        self.w = 1  # size of freq smoothing windows fun = 2*w+1
        self.alpha_s_ref = 0.9  # 平滑操作的 循环平均参数
        self.Nwin = 8  # 局部搜索的值
        self.Vwin = 15  # 观察的窗大小
        self.delta_s = 1.67  # =1/u(F^(-1)_(X^2;u)(1-0.01))  P(delta(k,l)>= delta_s |(H0(k,l))语音不存在) <0.01 语音不存在的条件下, (delta(k,l)>= delta_s)  的概率小于0.01
        self.Bmin = 1.66  # imrca 引入的平滑, The factor represents the bias of a minimum noise estimate
        self.delta_y = 4.6 # = -np.log(0.01)  P(delta_min(k,l) >= delta_y | H0(k,l)) < 0.01 语音不存在的条件下, (delta_min(k,l) >= delta_y)  的概率小于0.01
        self.delta_yt = 3  # = -np.log(0.05) P(delta_min_hat(k,l)>delta_yt |H0(k,l)) < 0.05  语音不存在的条件下, 估计的delta_min > delta_yt 的概率小于0.05
        self.alpha_d_ref = 0.85  # 语音存在条件概率计算时变平滑参数 alpha_d_hat(k,l)
        self.beta =(self.delta_yt-1-np.exp(-1)+np.exp(-self.delta_yt))/(self.delta_yt-1-3*np.exp(-1)+(self.delta_yt+2)*np.exp(-self.delta_yt))   #语音不存在的时候的补偿偏置

        # 3 Parameters of a Priori Probability for Signal-Absence Estimate   信号确实的先验概率估计参数
        self.alpha_xi_ref = 0.7  # 论文里面没有这个参数的更新, 用于平滑更新 xi(k,l)
        self.w_xi_local = 1
        self.w_xi_global = 15
        self.f_u = int(10e3)
        self.f_l = 50
        self.P_min = 0.005
        self.xi_lu_dB = -5
        self.xi_ll_dB = -10
        self.xi_gu_dB = -5
        self.xi_gl_dB = -10
        self.xi_fu_dB = -5
        self.xi_fl_dB = -10
        self.xi_mu_dB = 10
        self.xi_ml_dB = 0
        self.q_max = 0.998

        # 4  param of DD a priori SNR estimation
        self.alpha_eta_ref = 0.95
        self.eta_min_dB = -18

        # 5 Flags
        self.broad_flag = 1
        self.tone_flag = 1
        self.nonstat = "medium"  # "high"

        if (fs != self.Fs_ref):
            self.Fs = fs
            self.M = int(2 ** (np.round(np.log2(fs / self.Fs_ref * self.M_ref))))
            self.Mo = int(self.Mo_ref / self.M_ref * self.M)
            self.alpha_s = self.alpha_s_ref ** (self.M_ref / self.M * fs / self.Fs_ref)
            self.alpha_d = self.alpha_d_ref ** (self.M_ref / self.M * fs / self.Fs_ref)
            self.alpha_eta = self.alpha_eta_ref ** (self.M_ref / self.M * fs / self.Fs_ref)
            self.alpha_xi = self.alpha_xi_ref ** (self.M_ref / self.M * fs / self.Fs_ref)
        else:
            self.Fs = fs
            self.M = self.M_ref
            self.Mo = self.Mo_ref
            self.alpha_s = self.alpha_s_ref
            self.alpha_d = self.alpha_d_ref
            self.alpha_eta = self.alpha_eta_ref
            self.alpha_xi = self.alpha_xi_ref

        # 6 other param
        self.alpha_d_long = 0.99
        self.eta_min = 10 ** (self.eta_min_dB / 10)
        self.G_f = self.eta_min ** (0.5)

        # 7 window funtion
        self.win = np.hamming(self.M)
        self.win2 = self.win ** 2
        self.Mno = self.M - self.Mo
        self.W0 = self.win2[:self.Mno]

        for k in range(self.Mno, self.M, self.Mno):
            self.swin2 = self.lnshift(self.win2, k)
            self.W0 += self.swin2[:self.Mno]

        self.W0 = np.mean(self.W0) ** (0.5)
        self.win = self.win / self.W0
        self.Cwin = np.sum(np.square(self.win)) ** (0.5)
        self.win = self.win / self.Cwin

        # 8 smooth windows init  local_global serch???
        self.b = np.hanning(2 * (self.w +1)+ 1)[1:-1]
        self.b = self.b / np.sum(self.b)
        self.b_xi_local = np.hanning(2 * (self.w_xi_local+1) + 1)[1:-1]
        self.b_xi_local = self.b_xi_local / np.sum(self.b_xi_local)
        self.b_xi_global = np.hanning(2 * (self.w_xi_global+1) + 1)[1:-1]
        self.b_xi_global = self.b_xi_global / np.sum(self.b_xi_global)
        self.l_mod_lswitch = 0
        self.M21 = int(self.M / 2 + 1)

        self.k_u = int(np.round(self.f_u / self.Fs * self.M + 1))
        self.k_l = int(np.round(self.f_l / self.Fs * self.M + 1))
        self.k_u = np.min((self.k_u, self.M21))
        self.k2_local = int(np.round(500 / self.Fs * self.M + 1))
        self.k3_local = int(np.round(3500 / self.Fs * self.M + 1))

        self.eta_2term = 1
        self.xi = 0
        self.xi_frame = 0

        # 9 new version 3 flasg
        self.l_fnz = 1  #1
        self.l = 1
        self.fnz_flag = 0
        self.zero_thres = 1e-10

        # 10 stft_stream init&process
        self.prev_Mo_data = np.zeros(self.M, np.float)
        self.output_tmp = np.zeros(self.M+self.M,np.float)
        #self.exp_para = np.vectorize(self.expint)


        self.S =-1
        self.Smin =-1
        self.Smact =-1
        self.Smint =-1
        self.Smactt =-1
        self.lambda_dav_long =-1
        self.SW =-1
        self.SWt =-1
        self.St =-1
        self.lambda_dav =-1
        self.Sy =-1
        self.lambda_d = -1

        self.debug=0


    def denoise_process(self, Y, tmp):

        Ya2 = np.abs(Y) ** 2  # shape (M21,)
        if (self.l == self.l_fnz):
            self.lambda_d = Ya2.copy()  # shape (M21,)
        assert not isinstance(self.lambda_d ,int) ,f"error lambda_d {sys._getframe().f_lineno}"
        gamma = Ya2 / np.maximum(self.lambda_d, 1e-10)  # shape (M21,)
        eta = self.alpha_eta * self.eta_2term + (1 - self.alpha_eta) * np.maximum(gamma - 1, 0)
        eta = np.maximum(eta, self.eta_min)
        v = gamma * eta / (1 + eta)  # shape (M21,)
        # 2.1 smooth over freq
        Sf = np.convolve(self.b, Ya2)
        Sf = Sf[self.w: self.M21 + self.w]  # #shape (M21,)

        if (self.l == self.l_fnz):
            self.Sy = Ya2.copy()  # shape (M21,)
            self.S = Sf.copy()  # shape (M21,)
            self.St = Sf.copy()  # shape (M21,)
            self.lambda_dav = Ya2.copy()  # shape (M21,)
        else:
            assert not isinstance(self.S ,int) ,f"error S {sys._getframe().f_lineno}"
            self.S = self.alpha_s * self.S + (1 - self.alpha_s) * Sf  # shape (M21,)

        assert not isinstance(self.S, int), f"error S {sys._getframe().f_lineno}"
        if self.l < 14 + self.l_fnz:
            self.Smin = self.S.copy()
            self.Smact = self.S.copy()
        else:
            assert not  isinstance(self.Smin,int), f"error Smin {sys._getframe().f_lineno}"
            assert not isinstance(self.Smact, int), f"error Smact {sys._getframe().f_lineno}"
            self.Smin = np.minimum(self.Smin, self.S)  # shape (M21,)
            self.Smact = np.minimum(self.Smact, self.S)  # shape (M21,)

        I_f = np.logical_and(Ya2 < (self.delta_y * self.Bmin * self.Smin), self.S < (self.delta_s * self.Bmin * self.Smin))
        conv_I = np.convolve(self.b, I_f.astype(np.float))
        conv_I = conv_I[self.w:self.M21 + self.w]  # shape (M21,)
        assert not  isinstance(self.St,int), f"error St {sys._getframe().f_lineno}"
        Sft = self.St.copy()
        assert np.where(conv_I != 0).__len__() <2 , "idx error "
        idx = np.where(conv_I != 0)[0]
        if idx.size > 0:
            if self.w:
                conv_Y = np.convolve(self.b, I_f * Ya2)
                conv_Y = conv_Y[self.w:self.M21 + self.w]
                Sft[idx] = conv_Y[idx] / conv_I[idx]
            else:
                Sft[idx] = Ya2[idx]

        if self.l < 14 + self.l_fnz:
            self.St = self.S.copy()
            self.Smint = self.St.copy()  # Smin 的估计值
            self.Smactt = self.St.copy()  # Smact的估计值
        else:
            assert not isinstance(self.Smint ,int), f"error Smint {sys._getframe().f_lineno}"
            assert not isinstance(self.Smactt ,int), f"error Smactt {sys._getframe().f_lineno}"
            self.St = self.alpha_s * self.St + (1 - self.alpha_s) * Sft
            self.Smint = np.minimum(self.Smint, self.St)
            self.Smactt = np.minimum(self.Smactt, self.St)

        qhat = np.ones(self.M21, )
        phat = np.zeros(self.M21, )

        if self.nonstat == "low":
            gamma_mint = Ya2 / self.Bmin / np.maximum(self.Smin, 1e-10)
            zetat = self.S / self.Bmin / np.maximum(self.Smin, 1e-10)
        else:
            gamma_mint = Ya2 / self.Bmin / np.maximum(self.Smint, 1e-10)
            zetat = self.S / self.Bmin / np.maximum(self.Smint, 1e-10)

        condi1 = np.logical_and(gamma_mint > 1, gamma_mint < self.delta_yt)
        condi2 = np.logical_and(condi1, zetat < self.delta_s)
        assert np.where(condi2 != 0).__len__() < 2, "idx error "
        idx = np.where(condi2 != 0)[0]
        #### assert idx.size >0 , f"idx is 0 length, {sys._getframe().f_lineno}"
        qhat[idx] = (self.delta_yt - gamma_mint[idx]) / (self.delta_yt - 1)
        phat[idx] = 1 / (1 + qhat[idx] / (1 - qhat[idx]) * (1 + eta[idx]) * np.exp(-v[idx]))
        idx_phat = np.logical_or(gamma_mint >= self.delta_yt, zetat >= self.delta_s)
        phat[idx_phat] = 1
        alpha_dt = self.alpha_d + (1 - self.alpha_d) * phat
        assert not isinstance(self.lambda_dav ,int), f"error lambda_dav {sys._getframe().f_lineno}"
        self.lambda_dav = alpha_dt * self.lambda_dav + (1 - alpha_dt) * Ya2

        if (self.l < 14 + self.l_fnz):
            self.lambda_dav_long = self.lambda_dav.copy()
        else:
            assert not isinstance(self.lambda_dav_long ,int), f"error lambda_dav_long {sys._getframe().f_lineno}"
            alpha_dt_long = self.alpha_d_long + (1 - self.alpha_d_long) * phat
            self.lambda_dav_long = alpha_dt_long * self.lambda_dav_long + (1 - alpha_dt_long) * Ya2

        self.l_mod_lswitch += 1
        if self.l_mod_lswitch == self.Vwin:
            self.l_mod_lswitch = 0
            if self.l == self.Vwin - 1 + self.l_fnz:
                self.SW = np.tile(self.S, (self.Nwin, 1)).T  # shape (M21,Nwin)
                self.SWt = np.tile(self.St, (self.Nwin, 1)).T  # shape (M21,Nwin)
            else:
                assert not isinstance(self.SW ,int), f"error SW {sys._getframe().f_lineno}"
                assert not isinstance(self.SWt ,int), f"error SWt {sys._getframe().f_lineno}"
                self.SW = np.hstack((self.SW[:, 1:self.Nwin], (self.Smact).reshape((-1,1))))
                self.Smin = np.min(self.SW, axis=1)
                self.Smact = self.S.copy()
                self.SWt = np.hstack((self.SWt[:, 1:self.Nwin], (self.Smactt).reshape((-1,1))))
                self.Smint = np.min(self.SWt, axis=1)
                self.Smactt = self.St.copy()

        if self.nonstat == "high":
            self.lambda_d = self.beta * self.lambda_dav
        else:
            self.lambda_d =self.beta * self.lambda_dav

        self.xi = self.alpha_xi * self.xi + (1 - self.alpha_xi) * eta
        xi_local = np.convolve(self.xi, self.b_xi_local)
        xi_local = xi_local[self.w_xi_local:self.M21 + self.w_xi_local]
        xi_global = np.convolve(self.xi, self.b_xi_global)
        xi_global = xi_global[self.w_xi_global:self.M21 + self.w_xi_global]

        dxi_frame = self.xi_frame
        self.xi_frame = np.mean(self.xi[self.k_l:self.k_u])
        dxi_frame = self.xi_frame - dxi_frame
        xi_local_dB = 10 * np.log10(xi_local) if xi_local.all() > 0 else [-100] * self.M21
        xi_global_dB = 10 * np.log10(xi_global) if xi_global.all() > 0 else [-100] * self.M21
        xi_frame_dB = 10 * np.log10(self.xi_frame) if self.xi_frame > 0 else -100

        P_local = np.ones(self.M21)
        P_local[xi_local_dB <= self.xi_ll_dB] = self.P_min
        condi1 = np.logical_and(xi_local_dB > self.xi_ll_dB, xi_local_dB < self.xi_lu_dB)
        assert np.where(condi1 != 0).__len__() < 2, "idx error "
        idx = np.where(condi1 != 0)[0]
        ####assert idx.size > 0, f"idx is 0 length, {sys._getframe().f_lineno}"
        P_local[idx] = self.P_min + (xi_local_dB[idx] - self.xi_ll_dB) / (self.xi_lu_dB - self.xi_ll_dB) * (1 - self.P_min)

        P_global = np.ones(self.M21)
        P_global[xi_global_dB <= self.xi_gl_dB] = self.P_min
        condi2 = np.logical_and(xi_global_dB > self.xi_gl_dB, xi_global_dB < self.xi_gu_dB)
        assert np.where(condi2 != 0).__len__() < 2, "idx error "
        idx = np.where(condi2 != 0)[0]
        ####assert idx.size > 0, f"idx is 0 length, {sys._getframe().f_lineno}"
        P_global[idx] = self.P_min + (xi_global_dB[idx] - self.xi_gl_dB) / (self.xi_gu_dB - self.xi_gl_dB) * (1 - self.P_min)

        m_P_local = np.mean(P_local[2:(self.k2_local + self.k3_local - 2)])
        if m_P_local < 0.25:
            P_local[self.k2_local:self.k3_local] = self.P_min
        if self.tone_flag:
            if m_P_local < 0.5 and self.l > 120:
                condi1 = (self.lambda_dav_long[7:self.M21 - 8]) > (2.5 * (self.lambda_dav_long[9:self.M21 - 6]) + self.lambda_dav_long[5:self.M21 - 10])
                assert np.where(condi1 != 0).__len__() < 2, "idx error "
                idx = np.where(condi1 != 0)[0]
                if idx.size > 0:
                    idx1 = idx + 6
                    idx2 = idx + 7
                    idx3 = idx + 8
                    assert idx3.all() < self.M21, f"error idx3 {sys._getframe().f_lineno}"
                    P_local[idx1] = self.P_min
                    P_local[idx2] = self.P_min
                    P_local[idx3] = self.P_min

        if xi_frame_dB <= self.xi_fl_dB:
            P_frame = self.P_min
        elif dxi_frame >= 0:
            self.xi_m_dB = np.minimum(np.maximum(xi_frame_dB, self.xi_ml_dB), self.xi_mu_dB)
            P_frame = 1
        elif xi_frame_dB >= self.xi_m_dB + self.xi_fu_dB:
            assert self.xi_m_dB != -1, f"error xi_m_dB {sys._getframe().f_lineno}"
            P_frame = 1
        elif xi_frame_dB <= self.xi_m_dB + self.xi_fl_dB:
            assert self.xi_m_dB != -1, f"error xi_m_dB {sys._getframe().f_lineno}"
            P_frame = self.P_min
        else:
            assert self.xi_m_dB != -1, f"error xi_m_dB {sys._getframe().f_lineno}"
            P_frame = self.P_min + (xi_frame_dB - self.xi_m_dB - self.xi_fl_dB) / (self.xi_fu_dB - self.xi_fl_dB) * (1 - self.P_min)

        if self.broad_flag:
            q = 1 - P_global * P_local * P_frame
        else:
            q = 1 - P_local * P_frame
        q = np.minimum(q, self.q_max)

        gamma = Ya2 / np.maximum(self.lambda_d, 1e-10)
        eta = self.alpha_eta * self.eta_2term + (1 - self.alpha_eta) * np.maximum(gamma - 1, 0)
        eta = np.maximum(eta, self.eta_min)
        v = gamma * eta / (1 + eta)

        PH1 = np.zeros(self.M21)
        assert np.where(q < 0.9).__len__() < 2, "idx error "
        idx = np.where(q < 0.9)[0]
        #### assert idx.size > 0, f"idx is 0 length, {sys._getframe().f_lineno}"
        PH1[idx] = 1 / (1 + q[idx] / (1 - q[idx]) * (1 + eta[idx]) * np.exp(-v[idx]))

        GH1 = np.ones(self.M21)
        assert np.where(v >5).__len__() < 2, "idx error "
        idx = np.where(v > 5)[0]
        #### assert idx.size > 0, f"idx is 0 length, {sys._getframe().f_lineno}"
        GH1[idx] = eta[idx] / (1 + eta[idx])
        condi1 = np.logical_and(v <= 5, v > 0)
        assert np.where(condi1 != 0).__len__() < 2, "idx error "
        idx = np.where(condi1 != 0)[0]
        #### assert idx.size > 0, f"idx is 0 length, {sys._getframe().f_lineno}"
        # if(idx.size >0):
        #     GH1[idx] = eta[idx] / (1 + eta[idx]) * np.exp(0.5 * self.exp_para(v[idx]))
        GH1[idx] = eta[idx] / (1 + eta[idx]) * np.exp(0.5 * sc.expn(1,v[idx]))

        if self.tone_flag:
            lambda_d_global = self.lambda_d.copy()
            lambda_d_global_stack = np.vstack((lambda_d_global[3:self.M21 - 3], lambda_d_global[:self.M21 - 6], lambda_d_global[6:self.M21])).T
            lambda_d_global[3:self.M21 - 3] = np.min(lambda_d_global_stack, axis=1)
            assert not isinstance(self.Sy, int), f"error Sy {sys._getframe().f_lineno}"
            self.Sy = 0.8 * self.Sy + 0.2 * Ya2
            GH0 = self.G_f * (lambda_d_global / (self.Sy + 1e-10)) ** 0.5
        else:
            GH0 = self.G_f

        G = (GH1 ** (PH1)) * (GH0 ** (1 - PH1))
        self.eta_2term = (GH1 ** 2) * gamma

        X = np.hstack((np.zeros(3, ), G[3:self.M21 - 1] * Y[3:self.M21 - 1], 0))  # shape (M21,)
        return X


    @classmethod
    def read_data(cls, fin):
        sr, data = wavfile.read(fin)
        if data.dtype == np.int16:
            data = data * 1.0 / (2 ** 15)
        elif data.dtype == np.int32:
            data = data * 1.0 / (2 ** 31)
        elif data.dtype == np.float or data.dtype == np.double:
            data = data.astype(np.float)
        else:
            print("ERROR Unknow audio file data Type")
            sys.exit(-1)
        return sr, data

    def lnshift(self, x, t):
        szX = x.shape
        if szX[0] > 1:
            y = np.roll(x, t, axis=0)
        else:
            y = np.roll(x, t, axis=1)
        return y


    #   1 2 3 4 5 6
    #   0 1 2 3 4 5
    def stream_process(self, y):
        if self.debug==0:
            self.prev_Mo_data = y
        inputdata = np.hstack((self.prev_Mo_data,y))    #shape ( Mo+ y)

        for i in range(0, inputdata.shape[0] - self.M+1, self.Mno):
            tmp = inputdata[i:i + self.M]
            if (not self.fnz_flag and np.abs(y[0]) > self.zero_thres) or (self.fnz_flag and np.any(np.abs(y) > self.zero_thres)):
                self.fnz_flag = 1
                fft_data = np.fft.rfft(tmp * self.win,self.M)
                #X = self.denoise_process(fft_data, y)
                x = np.fft.irfft(fft_data,self.M)
                x = (self.Cwin ** 2) * self.win * x
                self.output_tmp[i:i + self.M] += x

            else:
                if not self.fnz_flag:
                    self.l_fnz += 1
                    print("l_fnz =", self.l_fnz)
            self.l +=1

        self.prev_Mo_data = inputdata[self.M:]
        output = self.output_tmp[:self.M]
        self.output_tmp=np.hstack((self.output_tmp[self.M:], [0]*self.M))

        return output

    #   1 1 2 3 4 5 6
    #   0 1 2 3 4 5
    def align_stream(self, y):  # 一直会多出一帧数据暂时没搞定, 原因是求出当前帧的数据, 需要用到N-1 和N+1 的信息来处理, 而且当前输出out其实就是prev的数据的结果
        # (prev, cur)1,1 -> 1(out) ; 1,2 ->1; 2,3 ->2 所以就多出一个1的输出结果了

        if self.debug==1:
            out = self.stream_process(y)
        else:
            out = self.stream_process(y)
            self.debug = 1
            out = self.stream_process(y)


            #out = self.stream_process(y)



        return out


    def expint(self, v):
        return si.quad(lambda t: np.exp(-t) / t, v, np.inf)[0]


    def librosa_outlineprocess(self, data):
        stft_data=librosa.stft(data,self.M, self.Mno, self.M, "hann")
        out_stft =np.zeros_like(stft_data)

        j=0
        for i in range(stft_data.shape[1]):
            tmp=data[j:j+self.M]
            if (tmp.shape[0] != self.M):
                pad= np.zeros(self.M-tmp.shape[0],np.float)
                tmp=np.hstack((tmp,pad))
            Y = stft_data[:,i]
            if (not self.fnz_flag and np.abs(tmp[0]) > self.zero_thres) or (self.fnz_flag and np.any(np.abs(tmp)) > self.zero_thres):
                self.fnz_flag = 1
                X=self.denoise_process(Y, tmp)
            else:
                if not self.fnz_flag:
                    self.l_fnz += 1
                X= np.zeros(self.M21, np.complex)
            self.l += 1
            j+=self.Mno
            out_stft[:,i] = X

        return librosa.istft(out_stft, self.Mno, self.M, "hann")

    def outline_process(self,data):   
        outputdata =np.zeros_like(data)
        out=np.zeros(self.M,np.float)
        Nframes=int((data.shape[0]-self.Mo)/self.Mno)
        j=0
        for i in range(Nframes):
            tmp=data[j:j+self.M]
            if (tmp.shape[0] != self.M):
                pad= np.zeros(self.M-tmp.shape[0],np.float)
                tmp=np.hstack((tmp,pad))

            Y= np.fft.rfft(tmp*self.win,self.M)
            if (not self.fnz_flag and np.abs(tmp[0]) > self.zero_thres) or (self.fnz_flag and np.any(np.abs(tmp)) > self.zero_thres):
                self.fnz_flag = 1
                X=self.denoise_process(Y, tmp)
                x = np.fft.irfft(X,self.M)
                x = (self.Cwin ** 2) * self.win * x
                out += x
            else:
                if not self.fnz_flag:
                    self.l_fnz += 1
            self.l += 1

            outputdata[j:j+self.Mno] = out[:self.Mno]
            j+=self.Mno
            out=np.hstack((out[self.Mno:], [0]*self.Mno))

        return outputdata


if __name__ == "__main__":
    import sys
    noise_file = r"./xxx.wav"
    out_file = r"./ooo.wav"

    sr, data = imcra.read_data(noise_file)
    outdata = np.zeros_like(data)
    imcra_demo = imcra(sr)

    # outdata=imcra_demo.librosa_outlineprocess(data)   # 已经做好复现matlab代码处理效果
    # wavfile.write(out_file, sr, outdata)

    outdata=imcra_demo.outline_process(data)   # 已经做好复现matlab代码处理效果
    wavfile.write(out_file, sr, outdata)
    print(imcra_demo.l)

    # imcra_demo = imcra(sr)          # 还没搞定流式处理, 会有bug多出一帧的声音, 无法实现当前帧进去给出当前帧的结果
    # for i in range(0, data.shape[0] - imcra_demo.M, imcra_demo.M):
    #     tmp_data = data[i:i + imcra_demo.M]
    #     out = imcra_demo.align_stream(tmp_data)
    #     outdata[i:i + imcra_demo.M] = out
    # # print(imcra_demo.l)
    # wavfile.write(out_file, sr, outdata)
