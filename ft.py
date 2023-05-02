from manim import *
from manim_tikz import Tikz
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
from manim_voiceover.services.azure import AzureService

FREQ_A = 440
FREQ_CX = 554.37
FREQ_E = 659.25


def plot_a(x):
    return np.sin(FREQ_A * x)


def plot_cx(x):
    return np.sin(FREQ_CX * x)


def plot_e(x):
    return np.sin(FREQ_E * x)


def compound(x):
    return plot_a(x) + plot_cx(x) + plot_e(x)

def dirac_delta(x):
    b = 10
    return np.exp(-(x/b)**2)/(b*np.sqrt(np.pi/2))

def compound_fft(x):
    return np.sqrt(np.pi/2) * (dirac_delta(x - FREQ_A) + dirac_delta(x - FREQ_CX) + dirac_delta(x - FREQ_E))

class Intro(VoiceoverScene):
    def construct(self):
        # self.set_speech_service(GTTSService(transcription_model='base'))
        self.set_speech_service(AzureService(prosody={'rate': '-20%'}))

        intro = Text("Quantum Fourier Transform", font_size=60).move_to(ORIGIN)
        subtitle = Text("by Petr Zahradník", font_size=40).next_to(intro, DOWN)
        with self.voiceover(text="In this video, I will try to explain the Quantum Fourier Transform.") as tracker:
            self.play(Write(intro))
            self.play(Write(subtitle))

        self.play(FadeOut(intro), FadeOut(subtitle))

        # title
        title = Title(
            r"Motivation",
            include_underline=False,
            font_size=40,
        )
        with self.voiceover(text="Let's start with a simple example.") as tracker:
            self.play(Write(title))

        # compound
        axes_compound = Axes(
            x_range=[0, 0.1, 1e-2],
            y_range=[-2.5, 2.5, 0.05],
            x_length=9,
            y_length=3,
            tips=False,
            axis_config={
                "include_ticks": False,
            },
        )
        plt_compound = axes_compound.plot(compound, color=WHITE)
        gr_compound = VGroup(axes_compound, plt_compound)

        with self.voiceover(text="Suppose we have a recording of a piano chord.") as tracker:
            self.play(Create(axes_compound), run_time=tracker.duration)

        self.add_sound("sounds/a-major.mp3")
        self.play(Create(plt_compound))
        self.play(
            gr_compound.animate.shift(UP * 2 + LEFT * 1),
            FadeOut(title),
        )

        # compound copies
        plt_compound_1 = axes_compound.plot(compound, color=WHITE)
        plt_compound_2 = axes_compound.plot(compound, color=WHITE)
        plt_compound_3 = axes_compound.plot(compound, color=WHITE)
        self.add(plt_compound_1, plt_compound_2, plt_compound_3)

        # A
        axes_a = Axes(
            x_range=[0, 0.1, 1e-2],
            y_range=[-1, 1, 0.05],
            x_length=9,
            y_length=1,
            tips=False,
            axis_config={
                "include_ticks": False,
            },
        )
        axes_a.next_to(axes_compound, DOWN, buff=0.3)
        plt_a = axes_a.plot(plot_a, color=BLUE)

        # C#
        axes_cx = Axes(
            x_range=[0, 0.1, 1e-2],
            y_range=[-1, 1, 0.05],
            x_length=9,
            y_length=1,
            tips=False,
            axis_config={
                "include_ticks": False,
            },
        )
        axes_cx.next_to(axes_a, DOWN, buff=0.3)
        plt_cx = axes_cx.plot(plot_cx, color=RED)

        # E
        axes_e = Axes(
            x_range=[0, 0.1, 1e-2],
            y_range=[-1, 1, 0.05],
            x_length=9,
            y_length=1,
            tips=False,
            axis_config={
                "include_ticks": False,
            },
        )
        axes_e.next_to(axes_cx, DOWN, buff=0.3)
        plt_e = axes_e.plot(plot_e, color=GREEN)

        # show axes
        with self.voiceover(text="And we would like to see the individual tones it is composed of.") as tracker:
            self.play(
                Create(axes_a),
                Create(axes_cx),
                Create(axes_e),
            )

        # split components
        with self.voiceover(text="The Fourier Transform allows us to decompose the sound into:") as tracker:
            self.play(
                Transform(plt_compound_1, plt_a),
                Transform(plt_compound_2, plt_cx),
                Transform(plt_compound_3, plt_e),
            )

        # show labels
        freq_a = Text("440 Hz", font_size=30).next_to(axes_a, RIGHT, buff=0.3)
        freq_cx = Text("554 Hz", font_size=30).next_to(axes_cx, RIGHT, buff=0.3)
        freq_e = Text("659 Hz", font_size=30).next_to(axes_e, RIGHT, buff=0.3)
        self.add_sound("sounds/440.mp3")
        self.play(Write(freq_a))
        self.wait(0.5)
        self.add_sound("sounds/554.mp3")
        self.play(Write(freq_cx))
        self.wait(0.5)
        self.add_sound("sounds/659.mp3")
        self.play(Write(freq_e))

        label_compound = Text("A maj", font_size=30).next_to(axes_compound, RIGHT, buff=0.3)
        label_a = Text("A", font_size=30).next_to(axes_a, RIGHT, buff=0.3)
        label_cx = Text("C#", font_size=30).next_to(axes_cx, RIGHT, buff=0.3)
        label_e = Text("E", font_size=30).next_to(axes_e, RIGHT, buff=0.3)
        with self.voiceover(text="This means that the original recording must have been the <bookmark mark='A' />A major chord!") as tracker:
            self.play(
                Transform(freq_a, label_a),
                Transform(freq_cx, label_cx),
                Transform(freq_e, label_e)
            )        
            self.wait_until_bookmark("A")
            self.play(Write(label_compound))

        self.wait(3)

class FT(VoiceoverScene):
    def construct(self):
        # self.set_speech_service(GTTSService(transcription_model='base'))
        self.set_speech_service(AzureService(transcription_model='base', prosody={'rate': '-20%'}))

        # title
        title = Title(
            r"Fourier Transform",
            include_underline=False,
            font_size=40,
        )

        # FT equation
        equation = MathTex(r"\hat{f}(\omega)", "=",  r"\int_{-\infty}^{\infty}", "f(t)", r"e^{-2\pi i \omega t}", "dt")

        with self.voiceover(text="More formally, Fourier Transform is a linear operator given by this definition.") as tracker:
            self.play(Write(title))
            self.play(Write(equation))

        # FT domain
        arrow = Arrow(start=2*LEFT, end=2*RIGHT, color=WHITE).next_to(equation, UP, buff=1)
        arrow_label = Text("Fourier Transform", font_size=25).next_to(arrow, UP, buff=0.2)
        domain_g = MathTex(r"f{{:}} \mathbb{R} \rightarrow \mathbb{C}").next_to(arrow, LEFT, buff=0.5)
        domain_ghat = MathTex(r"\hat{f}{{:}} \mathbb{R} \rightarrow \mathbb{C}").next_to(arrow, RIGHT, buff=0.5)

        with self.voiceover(text="It is a map between an integrable function in time domain to a function in frequency domain. Note that both input and output are complex-valued functions, but for simplicity, we will only be showing their magnitudes here.") as tracker:
            self.play(
                Write(domain_g),
                Create(arrow),
                Write(arrow_label),
                Write(domain_ghat),
            )

        self.wait()

        # function plot
        arrow_plot = Arrow(start=2*LEFT, end=2*RIGHT, color=WHITE).next_to(equation, DOWN, buff=1.5)
        axes = Axes(
            x_range=[0, 0.1, 0.01],
            y_range=[-2.5, 2.5, 0.05],
            x_length=4,
            y_length=2,
            tips=False,
            x_axis_config={
                "include_ticks": False,
            },
            y_axis_config={
                "include_ticks": False,
            },
        )
        axes.next_to(arrow_plot, LEFT, buff=0.3)
        plt = axes.plot(compound, color=WHITE)

        # transformed plot
        axes_transformed = Axes(
            x_range=[200, 700, 100],
            y_range=[0, 0.1, 0.01],
            x_length=4,
            y_length=2,
            tips=False,
            x_axis_config={
                "unit_size": 100,
                "exclude_origin_tick": True,
                "numbers_to_include": [FREQ_A, FREQ_CX, FREQ_E],
                "include_ticks": False,
            },
            y_axis_config={
                "include_ticks": False,
            },
        )
        axes_transformed.next_to(arrow_plot, RIGHT, buff=0.3)
        plt_transformed = axes_transformed.plot(compound_fft, color=WHITE)
        with self.voiceover(text="Taking f, a function of time and applying the <bookmark mark='A'/>Fourier Transform, we get f hat, a <bookmark mark='B'/>function of frequency.") as tracker:
            self.play(Create(axes))
            self.play(Create(plt))
            self.wait_until_bookmark("A")
            self.play(Create(arrow_plot))
            self.wait_until_bookmark("B")
            self.play(Create(axes_transformed))
            self.play(Create(plt_transformed))

        with self.voiceover(text="The peaks correspond to the frequencies which are present in the original signal. If we took the peak frequencies and played them together, we would get the original signal back.") as tracker:
            pass

        self.wait(3)

        # title
        title_d = Title(
            r"Discrete Fourier Transform",
            include_underline=False,
            font_size=40,
        )
        with self.voiceover(text="In practice, we are working with discrete samples of the signal, so we use the <bookmark mark='A'/>Discrete Fourier Transform instead.") as tracker:
            self.play(
                FadeOut(arrow_plot),
                FadeOut(axes),
                FadeOut(plt),
                FadeOut(axes_transformed),
                FadeOut(plt_transformed)
            )
            # self.wait_until_bookmark("A")
            self.play(Transform(title, title_d))

        # FT domain
        domain_g_d = MathTex(r"\mathbf{x}{{:}} \mathbb{C}^N").next_to(arrow, LEFT, buff=0.5)
        domain_ghat_d = MathTex(r"\hat{\mathbf{x}}{{:}} \mathbb{C}^N").next_to(arrow, RIGHT, buff=0.5)

        with self.voiceover(text="The input is now a vector of complex numbers, and the output is also a vector of complex numbers.") as tracker:
            self.play(
                Transform(domain_g, domain_g_d),
                Transform(domain_ghat, domain_ghat_d),
            )

        # FT equation
        equation_mid = MathTex(r"\hat{x}_k", "=", r"\int_{-\infty}^{\infty}", "x_n", r"e^{ -2\pi i k}", "dt")
        equation_d = MathTex(r"\hat{x}_k", "=", r"\sum_{n=0}^{N-1}", r"x_n", r"e^{-2\pi i \frac{kn}{N}}")
    
        with self.voiceover(text="We replace them in the definition and use a <bookmark mark='A'/>finite sum insted of the integral.") as tracker:
            self.play(TransformMatchingTex(equation, equation_mid), run_time=2)
            self.wait_until_bookmark("A")
            self.play(TransformMatchingTex(equation_mid, equation_d), run_time=2)

        with self.voiceover(text="The Discrete Fourier Transform can be seen as tranformation between the time basis and the frequency or Fourier basis.") as tracker:
            pass

        self.wait(3)

class DFT(VoiceoverScene):
    def construct(self):
        # self.set_speech_service(GTTSService(transcription_model='base'))
        self.set_speech_service(AzureService(transcription_model='base', prosody={'rate': '-20%'}))

        # title
        title = Title(
            r"Discrete Fourier Transform",
            include_underline=False,
            font_size=40,
        )
        self.add(title)

        # FT equation
        eq_dft = MathTex(r"\hat{x}_k", "=", r"\frac{1}{\sqrt{N}}", r"\sum_{n=0}^{N-1}", r"x_n", r"e^{-2\pi i \frac{kn}{N}}")
        with self.voiceover(text="The Discrete Fourier Transform is usually written in a normalized form.") as tracker:
            self.play(Write(eq_dft))

        # linear map
        eq_dft_copy = MathTex(r"\hat{x}_k", "=", r"\frac{1}{\sqrt{N}}", r"\sum_{n=0}^{N-1}", r"x_n", r"e^{-2\pi i \frac{kn}{N}}").move_to(eq_dft.get_center())
        self.add(eq_dft_copy)
        eq_lin = MathTex(r"\hat{x}_k", "=", r"\mathbf{F}", r"x_k").next_to(eq_dft, DOWN, buff=1.5)
        with self.voiceover(text="Since we know it is a linear map, we can <bookmark mark='A'/>rewrite the Transform as a matrix multiplication.") as tracker:
            # self.wait_until_bookmark("A")
            self.play(TransformMatchingTex(eq_dft_copy, eq_lin))

        # hide all
        self.play(FadeOut(eq_dft), FadeOut(eq_dft_copy))

        # DFT matrix
        vandermonde = MathTex(r"\mathbf{F_N}", "=", r"\frac{1}{\sqrt{N}}", r"\begin{bmatrix} 1 & 1 & 1 & \dots & 1 \\ 1 & \omega & \omega^2 & \dots & \omega^{N-1} \\ 1 & \omega^2 & \omega^4 & \dots & \omega^{2(N-1)} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & \omega^{N-1} & \omega^{2(N-1)} & \dots & \omega^{(N-1)(N-1)} \end{bmatrix}").move_to(ORIGIN + 0.5*UP)
        F = eq_lin[2].copy()
        self.add(F)
        with self.voiceover(text="The Fourier matrix F of size N is a special kind of Vandermonde matrix.") as tracker:
            self.play(Transform(F, vandermonde[0]))
            self.play(Write(vandermonde[1:]))

        # omega definition
        omega = Tex(r"where $\omega = e^{-2\pi i \frac{1}{N}}$").next_to(vandermonde, DOWN, buff=0.3).shift(5*RIGHT)
        with self.voiceover(text="Omega here is the Nth root of unity.") as tracker:
            self.play(Write(omega))

        self.wait(3)

        with self.voiceover(text="Remember we normalized the matrix.") as tracker:
            self.play(Indicate(vandermonde[2], run_time=tracker.duration))

        # circumscribe unitary
        with self.voiceover(text="The Discrete Fourier Transform is therefore a unitary operator.") as tracker:
            self.play(Circumscribe(eq_lin[2], fade_out=True, run_time=tracker.duration))

        self.wait()

        with self.voiceover(text="Do you already see where this is going?", style="hopeful") as tracker:
            pass

        self.wait(3)

class QFT(VoiceoverScene):
    def construct(self):
        # self.set_speech_service(GTTSService(transcription_model='base'))
        self.set_speech_service(AzureService(transcription_model='base', prosody={'rate': '-20%'}))

        tex_template = TexTemplate()
        tex_template.add_to_preamble(r"\usepackage{physics}")

        # title
        title = Title(
            r"Quantum Fourier Transform",
            include_underline=False,
            font_size=40,
        )
        with self.voiceover(text="Let's now finally move to the Quantum Fourier Transform.") as tracker:
            self.play(Write(title))

        # 2D vandermonde
        vandermonde_2d = MathTex(r"\mathbf{F}_2", "=", r"\frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & \omega \end{bmatrix}").move_to(ORIGIN + UP)
        with self.voiceover(text="We start with the 2 by 2 Fourier matrix.") as tracker:
            self.play(Write(vandermonde_2d))

        self.wait()

        # 2D fourier matrix
        fourier_omega = MathTex(r"\mathbf{F}_2", "=", r"\frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & e^{-\pi i} \end{bmatrix}").move_to(vandermonde_2d.get_center())
        fourier_2d = MathTex(r"\mathbf{F}_2", "=", r"\frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}").move_to(vandermonde_2d.get_center())
        with self.voiceover(text="Does it look familiar to you?") as tracker:
            self.play(TransformMatchingTex(vandermonde_2d, fourier_omega, run_time=1))
            self.wait(0.5)
            self.play(TransformMatchingTex(fourier_omega, fourier_2d, run_time=1))

        self.wait(2)

        # equality to hadamard
        hadamard = MathTex(r"\mathbf{F}_2", "=", r"\frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}", "=", r"\mathbf{H}").move_to(fourier_2d.get_center())
        with self.voiceover(text="It is of course the <bookmark mark='A'/>Hadamard matrix!", style="excited") as tracker:
            self.wait_until_bookmark("A")
            self.play(TransformMatchingTex(fourier_2d, hadamard))

        self.wait(2)

        # state map
        state = MathTex(r"\alpha \ket{0} + \beta \ket{1} \mapsto \hat{\alpha} \ket{0} + \hat{\beta} \ket{1}", tex_template=tex_template).next_to(hadamard, DOWN, buff=1.5)
        with self.voiceover(text="The Hadamard actually performs the Fourier Transform on the amplitudes of computational basis states.") as tracker:
            self.play(Write(state))

        with self.voiceover(text="Note well that a single gate applied on a single qubit actually performs the transform on two amplitudes at once.") as tracker:
            pass

        self.wait(3)

class QFT_diagram(VoiceoverScene):
    def construct(self):
        # self.set_speech_service(GTTSService(transcription_model='base'))
        self.set_speech_service(AzureService(transcription_model='base', prosody={'rate': '-20%'}))

        tex_template = TexTemplate()
        tex_template.add_to_preamble(r"\usepackage{physics}")
        tex_template.add_to_preamble(r"\usepackage{tikz}\usetikzlibrary{quantikz2}")

        # title
        title = Title(
            r"Quantum Fourier Transform",
            include_underline=False,
            font_size=40,
        )
        with self.voiceover(text="Now that we know how the unitary operator works, let's see how the quantum circuits look like.") as tracker:
            self.add(title)

        # 1D circuit
        circuit_1d = Tikz(r"""
            \node[scale=1.5] {\begin{quantikz}[color=white,draw=white,transparent]
                & \gate{H} &
            \end{quantikz}};""",
            packages=["tikz"],
            libraries=["quantikz2"],
        )
        with self.voiceover(text="A single Hadamard performs the Quantum Fourier Transform on two amplitudes.") as tracker:
            self.play(Create(circuit_1d))

        self.wait()

        # 2D circuit
        circuit_2d = Tikz(r"""
            \node[scale=1.5] {\begin{quantikz}[color=white,draw=white,transparent]
                \lstick{\ket{x_1}} & \gate{H} & \gate{UROT_2} &          & \swap{1} & \\
                \lstick{\ket{x_2}} &          & \ctrl{-1}     & \gate{H} & \targX{} &
            \end{quantikz}};""",
            packages=["tikz"],
            libraries=["quantikz2"],
        )
        with self.voiceover(text="The 2-qubit circuit is a bit more complicated.") as tracker:
            self.play(Transform(circuit_1d, circuit_2d))

        self.wait()

        urot = MathTex(r"UROT_k = \begin{bmatrix} 1 & 0 \\ 0 & e^{\frac{-2\pi i}{2^k}} \end{bmatrix}").next_to(circuit_2d, DOWN, buff=1)
        with self.voiceover(text="The UROT gate rotates the phase of the qubit.", ssml="""The <phoneme alphabet="ipa" ph="juˈ.ɹɑt">UROT</phoneme> gate rotates the phase of the qubit.""") as tracker:
            self.play(Write(urot))

        with self.voiceover(text="It is sometimes called the phase shift gate. We use it here as a controlled gate to transfer the phase.") as tracker:
            pass

        with self.voiceover(text="Finally, we need to swap the output qubits and the Fourier Transform is finished.") as tracker:
            self.play(FadeOut(urot))

        self.wait()

        # 3D circuit
        circuit_3d = Tikz(r"""
            \node[scale=1.5] {\begin{quantikz}[color=white,draw=white,transparent]
                \lstick{\ket{x_1}} & \gate{H} & \gate{UROT_2} & \gate{UROT_3} &          &               &          & \swap{2} & \\
                \lstick{\ket{x_2}} &          & \ctrl{-1}     &               & \gate{H} & \gate{UROT_2} &          &          & \\
                \lstick{\ket{x_3}} &          &               & \ctrl{-2}     &          & \ctrl{-1}     & \gate{H} & \targX{} & \\
            \end{quantikz}};""",
            packages=["tikz"],
            libraries=["quantikz2"],
        )

        # 4D circuit
        circuit_4d = Tikz(r"""
            \node[scale=1.5] {\begin{quantikz}[color=white,draw=white,transparent]
                \lstick{\ket{x_1}} & \gate{H} & \gate{UROT_2} & \gate{UROT_3} & \gate{UROT_4} &          &               &               &          &               &          & \swap{3} &          & \\
                \lstick{\ket{x_2}} &          & \ctrl{-1}     &               &               & \gate{H} & \gate{UROT_2} & \gate{UROT_3} &          &               &          &          & \swap{1} & \\
                \lstick{\ket{x_3}} &          &               & \ctrl{-2}     &               &          & \ctrl{-1}     &               & \gate{H} & \gate{UROT_2} &          &          & \targX{} & \\
                \lstick{\ket{x_4}} &          &               &               & \ctrl{-3}     &          &               & \ctrl{-2}     &          & \ctrl{-1}     & \gate{H} & \targX{} &          &
            \end{quantikz}};""",
            packages=["tikz"],
            libraries=["quantikz2"],
        )

        with self.voiceover(text="The 3-qubit circuit repeats the same structure and performs the transform on 8 amplitudes.") as tracker:
            self.play(Transform(circuit_1d, circuit_3d))

        self.wait()

        brace_vertical = Brace(circuit_4d, direction=LEFT)
        brace_vertical_text = MathTex(r"\log N").next_to(brace_vertical, LEFT)
        brace_horizontal = Brace(circuit_4d, direction=DOWN)
        brace__horizontal_text = MathTex(r"\log^2 N").next_to(brace_horizontal, DOWN)
        with self.voiceover(text="The pattern continues on. We see that if we can encode the input vector into N amplitudes, only <bookmark mark='A'/>logarithm of N qubits are needed. The number of gates is <bookmark mark='B'/>quadratic but this is still an exponential speedup compared to the classical Fourier Transform!") as tracker:
            self.play(Transform(circuit_1d, circuit_4d))
            self.wait_until_bookmark("A")
            self.play(FadeIn(brace_vertical), Write(brace_vertical_text))
            self.wait_until_bookmark("B")
            self.play(FadeIn(brace_horizontal), Write(brace__horizontal_text))

        with self.voiceover(text="The best known classical algorithm, the Fast Fourier Transform runs in N log N time.") as tracker:
            pass

        self.wait(3)

class QFT_demo(VoiceoverScene):
    def construct(self):
        tex_template = TexTemplate()
        tex_template.add_to_preamble(r"\usepackage{physics}")
        # self.set_speech_service(GTTSService(transcription_model='base'))
        self.set_speech_service(AzureService(transcription_model='base', prosody={'rate': '-20%'}))

        # arrow
        arrow = Arrow(start=0.5*UP, end=DOWN, buff=0)
        arrow_text = Text("Quantum Fourier Transform", font_size=20).next_to(arrow, RIGHT)

        # input_label
        input_label_0000 = MathTex(r"0000", tex_template=tex_template)
        input_label_0001 = MathTex(r"0001", tex_template=tex_template).next_to(input_label_0000, RIGHT)
        input_label_0010 = MathTex(r"0010", tex_template=tex_template).next_to(input_label_0001, RIGHT)
        input_label_0011 = MathTex(r"0011", tex_template=tex_template).next_to(input_label_0010, RIGHT)
        input_label_0100 = MathTex(r"0100", tex_template=tex_template).next_to(input_label_0011, RIGHT)
        input_label_0101 = MathTex(r"0101", tex_template=tex_template).next_to(input_label_0100, RIGHT)
        input_label_0110 = MathTex(r"0110", tex_template=tex_template).next_to(input_label_0101, RIGHT)
        input_label_0111 = MathTex(r"0111", tex_template=tex_template).next_to(input_label_0110, RIGHT)
        input_label_1000 = MathTex(r"1000", tex_template=tex_template).next_to(input_label_0111, RIGHT)
        input_label_1001 = MathTex(r"1001", tex_template=tex_template).next_to(input_label_1000, RIGHT)
        input_label_1010 = MathTex(r"1010", tex_template=tex_template).next_to(input_label_1001, RIGHT)
        input_label_1011 = MathTex(r"1011", tex_template=tex_template).next_to(input_label_1010, RIGHT)
        input_label_1100 = MathTex(r"1100", tex_template=tex_template).next_to(input_label_1011, RIGHT)
        input_label_1101 = MathTex(r"1101", tex_template=tex_template).next_to(input_label_1100, RIGHT)
        input_label_1110 = MathTex(r"1110", tex_template=tex_template).next_to(input_label_1101, RIGHT)
        input_label_1111 = MathTex(r"1111", tex_template=tex_template).next_to(input_label_1110, RIGHT)
        input_label = VGroup(
            input_label_0000, input_label_0001, input_label_0010, input_label_0011, input_label_0100, input_label_0101, input_label_0110, input_label_0111,
            input_label_1000, input_label_1001, input_label_1010, input_label_1011, input_label_1100, input_label_1101, input_label_1110, input_label_1111
        )
        input_label.scale(0.6).next_to(arrow, UP, buff=2)
        with self.voiceover(text="Finally, let's see an example of the Quantum Fourier Transform of N <bookmark mark='A'/>equals 16, with 4 qubits.") as tracker:
            self.wait_until_bookmark("A")
            self.play(Write(input_label))

        # input
        input_0000 = MathTex(r"1", tex_template=tex_template).next_to(input_label_0000, DOWN)
        input_0001 = MathTex(r"0", tex_template=tex_template).next_to(input_label_0001, DOWN)
        input_0010 = MathTex(r"0", tex_template=tex_template).next_to(input_label_0010, DOWN)
        input_0011 = MathTex(r"0", tex_template=tex_template).next_to(input_label_0011, DOWN)
        input_0100 = MathTex(r"1", tex_template=tex_template).next_to(input_label_0100, DOWN)
        input_0101 = MathTex(r"0", tex_template=tex_template).next_to(input_label_0101, DOWN)
        input_0110 = MathTex(r"0", tex_template=tex_template).next_to(input_label_0110, DOWN)
        input_0111 = MathTex(r"0", tex_template=tex_template).next_to(input_label_0111, DOWN)
        input_1000 = MathTex(r"1", tex_template=tex_template).next_to(input_label_1000, DOWN)
        input_1001 = MathTex(r"0", tex_template=tex_template).next_to(input_label_1001, DOWN)
        input_1010 = MathTex(r"0", tex_template=tex_template).next_to(input_label_1010, DOWN)
        input_1011 = MathTex(r"0", tex_template=tex_template).next_to(input_label_1011, DOWN)
        input_1100 = MathTex(r"1", tex_template=tex_template).next_to(input_label_1100, DOWN)
        input_1101 = MathTex(r"0", tex_template=tex_template).next_to(input_label_1101, DOWN)
        input_1110 = MathTex(r"0", tex_template=tex_template).next_to(input_label_1110, DOWN)
        input_1111 = MathTex(r"0", tex_template=tex_template).next_to(input_label_1111, DOWN)
        input = VGroup(
            input_0000, input_0001, input_0010, input_0011, input_0100, input_0101, input_0110, input_0111,
            input_1000, input_1001, input_1010, input_1011, input_1100, input_1101, input_1110, input_1111
        )
        input_notation = MathTex(r"\mathbf{x}\colon", tex_template=tex_template).next_to(input_0000, LEFT, buff=0.5)
        with self.voiceover(text="Our input will we this vector repeating the same pattern four times") as tracker:
            self.play(Write(input_notation))
            self.play(Write(input))

        # input qubits
        input_qubits = MathTex(r"\frac{1}{2}(", r"\ket{0000}", "+", r"\ket{0100}", "+", r"\ket{1000}", "+", r"\ket{1100}", ")", tex_template=tex_template).next_to(input, DOWN)
        input_qubits_2 = MathTex(r"\frac{1}{2}(", r"\ket{0001}", "+", r"\ket{0101}", "+", r"\ket{1001}", r"+", r"\ket{1101}", ")", tex_template=tex_template).next_to(input, DOWN)
        input_qubits_3 = MathTex(r"\frac{1}{2}(", r"\ket{0010}", "+", r"\ket{0110}", "+", r"\ket{1010}", r"+", r"\ket{1110}", ")", tex_template=tex_template).next_to(input, DOWN)
        input_qubits_4 = MathTex(r"\frac{1}{2}(", r"\ket{0011}", "+", r"\ket{0111}", "+", r"\ket{1011}", r"+", r"\ket{1111}", ")", tex_template=tex_template).next_to(input, DOWN)
        self.play(Write(input_qubits))

        with self.voiceover(text="Let's now apply the <bookmark mark='A'/>Quantum Fourier Transform.") as tracker:
            self.wait_until_bookmark("A")
            self.play(Write(arrow), Write(arrow_text))

        # output qubits
        output_qubits = MathTex(r"\frac{1}{2}(\ket{0000} + \ket{0100} + \ket{1000} + \ket{1100})", tex_template=tex_template).next_to(arrow, DOWN)
        with self.voiceover(text="We will get <bookmark mark='A'/>this superposition as the output.") as tracker:
            self.wait_until_bookmark("A")
            self.play(Write(output_qubits))

        # output
        output_0000 = MathTex(r"1", tex_template=tex_template)
        output_0001 = MathTex(r"0", tex_template=tex_template).next_to(output_0000, RIGHT, buff=0.5)
        output_0010 = MathTex(r"0", tex_template=tex_template).next_to(output_0001, RIGHT, buff=0.5)
        output_0011 = MathTex(r"0", tex_template=tex_template).next_to(output_0010, RIGHT, buff=0.5)
        output_0100 = MathTex(r"1", tex_template=tex_template).next_to(output_0011, RIGHT, buff=0.5)
        output_0101 = MathTex(r"0", tex_template=tex_template).next_to(output_0100, RIGHT, buff=0.5)
        output_0110 = MathTex(r"0", tex_template=tex_template).next_to(output_0101, RIGHT, buff=0.5)
        output_0111 = MathTex(r"0", tex_template=tex_template).next_to(output_0110, RIGHT, buff=0.5)
        output_1000 = MathTex(r"1", tex_template=tex_template).next_to(output_0111, RIGHT, buff=0.5)
        output_1001 = MathTex(r"0", tex_template=tex_template).next_to(output_1000, RIGHT, buff=0.5)
        output_1010 = MathTex(r"0", tex_template=tex_template).next_to(output_1001, RIGHT, buff=0.5)
        output_1011 = MathTex(r"0", tex_template=tex_template).next_to(output_1010, RIGHT, buff=0.5)
        output_1100 = MathTex(r"1", tex_template=tex_template).next_to(output_1011, RIGHT, buff=0.5)
        output_1101 = MathTex(r"0", tex_template=tex_template).next_to(output_1100, RIGHT, buff=0.5)
        output_1110 = MathTex(r"0", tex_template=tex_template).next_to(output_1101, RIGHT, buff=0.5)
        output_1111 = MathTex(r"0", tex_template=tex_template).next_to(output_1110, RIGHT, buff=0.5)
        output = VGroup(
            output_0000, output_0001, output_0010, output_0011, output_0100, output_0101, output_0110, output_0111,
            output_1000, output_1001, output_1010, output_1011, output_1100, output_1101, output_1110, output_1111
        )
        output.next_to(output_qubits, DOWN)
        output_notation = MathTex(r"\mathbf{\hat{x}}\colon", tex_template=tex_template).next_to(output_0000, LEFT, buff=0.5)

        # output frequencies
        output_freq_0 = MathTex(r"0", tex_template=tex_template, font_size=20).next_to(output_0000, DOWN)
        output_freq_1 = MathTex(r"1", tex_template=tex_template, font_size=20).next_to(output_0001, DOWN)
        output_freq_2 = MathTex(r"2", tex_template=tex_template, font_size=20).next_to(output_0010, DOWN)
        output_freq_3 = MathTex(r"3", tex_template=tex_template, font_size=20).next_to(output_0011, DOWN)
        output_freq_4 = MathTex(r"4", tex_template=tex_template, font_size=20).next_to(output_0100, DOWN)
        output_freq_5 = MathTex(r"5", tex_template=tex_template, font_size=20).next_to(output_0101, DOWN)
        output_freq_6 = MathTex(r"6", tex_template=tex_template, font_size=20).next_to(output_0110, DOWN)
        output_freq_7 = MathTex(r"7", tex_template=tex_template, font_size=20).next_to(output_0111, DOWN)
        output_freq_8 = MathTex(r"8", tex_template=tex_template, font_size=20).next_to(output_1000, DOWN)
        output_freq_9 = MathTex(r"9", tex_template=tex_template, font_size=20).next_to(output_1001, DOWN)
        output_freq_10 = MathTex(r"10", tex_template=tex_template, font_size=20).next_to(output_1010, DOWN)
        output_freq_11 = MathTex(r"11", tex_template=tex_template, font_size=20).next_to(output_1011, DOWN)
        output_freq_12 = MathTex(r"12", tex_template=tex_template, font_size=20).next_to(output_1100, DOWN)
        output_freq_13 = MathTex(r"13", tex_template=tex_template, font_size=20).next_to(output_1101, DOWN)
        output_freq_14 = MathTex(r"14", tex_template=tex_template, font_size=20).next_to(output_1110, DOWN)
        output_freq_15 = MathTex(r"15", tex_template=tex_template, font_size=20).next_to(output_1111, DOWN)
        output_freq = VGroup(
            output_freq_0, output_freq_1, output_freq_2, output_freq_3, output_freq_4, output_freq_5, output_freq_6, output_freq_7,
            output_freq_8, output_freq_9, output_freq_10, output_freq_11, output_freq_12, output_freq_13, output_freq_14, output_freq_15
        )
        frequency = Text("freq:", font_size=20).next_to(output_freq_0, LEFT, buff=0.5)
        
        with self.voiceover(text="Given enough measurements we would be able to obtain <bookmark mark='A'/>these frequencies uniformly.") as tracker:
            self.play(Write(frequency))
            self.play(Write(output_freq))
            self.wait_until_bookmark("A")
            self.play(Write(output_notation))
            self.play(Write(output))

        self.wait(3)

        with self.voiceover(text="We can easily verify that the output is correct and same as the clasical Fourier Transform.") as tracker:
            pass

        with self.voiceover(text="Let us try to shift the original sequence, since the frequency did not change, there is no change in the output of the transform either.") as tracker:
            self.play(CyclicReplace(*input))
            self.play(TransformMatchingTex(input_qubits, input_qubits_2))
            self.play(Indicate(arrow))
            self.play(Indicate(output, scale_factor=1.05))
            self.wait()
            self.play(CyclicReplace(*input))
            self.play(TransformMatchingTex(input_qubits_2, input_qubits_3))
            self.play(Indicate(arrow))
            self.play(Indicate(output, scale_factor=1.05))
            self.wait()
            self.play(CyclicReplace(*input))
            self.play(TransformMatchingTex(input_qubits_3, input_qubits_4))
            self.play(Indicate(arrow))
            self.play(Indicate(output, scale_factor=1.05))

        self.wait()

        with self.voiceover(text="And this demonstration concludes our video. I hope you enjoyed it.", style="friendly") as tracker:
            pass

        self.play(
            *[FadeOut(mob)for mob in self.mobjects]
        )

        self.wait()