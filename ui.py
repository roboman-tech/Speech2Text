"""
NPC Dialogue Engine UI.
Panels: Live Caption/Dialogue History, User Context, Generated Answer.
"""

import tkinter as tk
from tkinter import font as tkfont

# Style constants — warm dark theme
BG = "#0f0f12"
BG_PANEL = "#18181c"
BG_INPUT = "#1e1e24"
FG = "#e8e8ed"
FG_DIM = "#8888a0"
ACCENT = "#3b82f6"
ACCENT_HOVER = "#60a5fa"
SUCCESS = "#22c55e"
LIVE = "#ef4444"
BORDER = "#2a2a32"
RADIUS = 2


def _make_label(parent, text: str, bg=BG, fg=FG_DIM, size=9):
    """Create a panel section label."""
    return tk.Label(parent, text=text, font=("Segoe UI", size), fg=fg, bg=bg)


def _make_panel(parent, title: str, icon: str = "") -> tuple:
    """Create a framed panel with header. Returns (content_frame, inner_widget_frame)."""
    frame = tk.Frame(parent, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
    header = tk.Frame(frame, bg=BG_PANEL)
    header.pack(fill=tk.X, padx=8, pady=(6, 2))
    lbl = tk.Label(header, text=f"{icon} {title}".strip(), font=("Segoe UI", 10, "bold"), fg=FG, bg=BG_PANEL)
    lbl.pack(anchor=tk.W)
    content = tk.Frame(frame, bg=BG_PANEL, padx=6, pady=4)
    content.pack(fill=tk.BOTH, expand=True)
    return frame, content


def _make_text(parent, height: int, readonly: bool = False, **kw):
    """Create a Text widget with scrollbar."""
    frame = tk.Frame(parent, bg=BG_PANEL)
    t = tk.Text(
        frame,
        font=("Segoe UI", 11),
        fg=FG,
        bg=BG_INPUT,
        wrap=tk.WORD,
        insertbackground=FG,
        selectbackground=ACCENT,
        selectforeground=FG,
        relief=tk.FLAT,
        padx=10,
        pady=8,
        height=height,
        **kw,
    )
    sb = tk.Scrollbar(frame, command=t.yview, bg=BG_PANEL, troughcolor=BORDER, activebackground=FG_DIM)
    t.configure(yscrollcommand=sb.set)
    t.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sb.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 2))
    if readonly:
        t.config(state=tk.DISABLED)
    return t, frame


class CaptionWindow:
    def __init__(
        self,
        width=1080,
        height=720,
        font_size=16,
        bg=BG,
        fg=FG,
        system_available=True,
        diarize_available=True,
        diarize_initial=False,
        interview_mode=True,
    ):
        self.root = tk.Tk()
        self.root.title("NPC Dialogue Engine")
        self.root.configure(bg=bg)
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(not interview_mode)  # Normal window in interview mode
        self.root.attributes("-alpha", 0.97)
        self.width = width
        self.height = height
        self.font_size = font_size
        self.bg = bg
        self.fg = fg
        self._interview_mode = interview_mode
        self._close_cb = None
        self._mode_change_cb = None
        self._stop_cb = None
        self._diarize_change_cb = None
        self._generate_cb = None
        self._save_chat_cb = None
        self._clear_cb = None
        self._transcriber_apply_cb = None
        self._system_available = system_available
        self._capturing = True
        self._diarize_available = diarize_available
        self._diarize_initial = diarize_initial
        self._last_speaker = None
        self._preview_mark = None
        self._setup_ui()
        if not interview_mode:
            self._make_draggable()

    def _setup_ui(self):
        self.root.configure(bg=BG)
        self.frame = tk.Frame(self.root, bg=BG, padx=12, pady=8)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # --- Title bar ---
        title_frame = tk.Frame(self.frame, bg=BG)
        title_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Label(title_frame, text="NPC Dialogue Engine", font=("Segoe UI", 14, "bold"), fg=FG, bg=BG).pack(side=tk.LEFT)
        self._status_dot = tk.Canvas(title_frame, width=10, height=10, bg=BG, highlightthickness=0)
        self._status_dot.pack(side=tk.RIGHT, padx=(8, 0))
        self._status_dot.create_oval(1, 1, 9, 9, fill=LIVE, outline="")
        self._status_label = tk.Label(title_frame, text="LIVE", font=("Segoe UI", 9, "bold"), fg=LIVE, bg=BG)
        self._status_label.pack(side=tk.RIGHT)

        # --- Toolbar ---
        toolbar = tk.Frame(self.frame, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
        toolbar.pack(fill=tk.X, pady=(0, 8))
        inner = tk.Frame(toolbar, bg=BG_PANEL, padx=10, pady=6)
        inner.pack(fill=tk.X)
        tk.Label(inner, text="Capture:", font=("Segoe UI", 9), fg=FG_DIM, bg=BG_PANEL).pack(side=tk.LEFT, padx=(0, 6))
        default_mode = "system" if self._system_available else "mic"
        self._mode_var = tk.StringVar(value=default_mode)
        modes = ["system", "both"] if self._system_available else []
        modes.append("mic")
        self._mode_menu = tk.OptionMenu(inner, self._mode_var, *modes, command=self._on_mode_change)
        self._mode_menu.config(font=("Segoe UI", 10), bg=BG_INPUT, fg=FG, highlightthickness=0, anchor="w")
        self._mode_menu.pack(side=tk.LEFT)
        tk.Label(inner, text="Device:", font=("Segoe UI", 9), fg=FG_DIM, bg=BG_PANEL).pack(side=tk.LEFT, padx=(12, 4))
        self._device_var = tk.StringVar(value="auto")
        for opt in ["auto", "gpu", "cpu"]:
            tk.Radiobutton(inner, text=opt.upper(), variable=self._device_var, value=opt, font=("Segoe UI", 8),
                           fg=FG_DIM, bg=BG_PANEL, selectcolor=BG_INPUT, activebackground=BG_PANEL,
                           activeforeground=FG, command=self._on_transcriber_settings_changed).pack(side=tk.LEFT, padx=2)
        tk.Label(inner, text="Model:", font=("Segoe UI", 9), fg=FG_DIM, bg=BG_PANEL).pack(side=tk.LEFT, padx=(12, 4))
        self._model_var = tk.StringVar(value="small.en")
        models = ["base.en", "small.en", "medium.en", "turbo"]
        self._model_menu = tk.OptionMenu(inner, self._model_var, *models)
        self._model_menu.config(font=("Segoe UI", 9), bg=BG_INPUT, fg=FG, highlightthickness=0, anchor="w")
        self._model_menu.pack(side=tk.LEFT)
        tk.Button(inner, text="Apply", font=("Segoe UI", 9), bg=ACCENT, fg=FG, relief=tk.FLAT, padx=8, pady=2,
                  command=self._apply_transcriber_settings, activebackground=ACCENT_HOVER, activeforeground=FG).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(inner, text="Clear", font=("Segoe UI", 9), bg=BG_INPUT, fg=FG, relief=tk.FLAT, padx=10, pady=2, command=self.clear_history,
                  activebackground=BORDER, activeforeground=FG).pack(side=tk.LEFT, padx=(12, 0))
        if self._interview_mode:
            tk.Button(
                inner, text="Save chat", font=("Segoe UI", 9), bg=BG_INPUT, fg=FG,
                relief=tk.FLAT, padx=10, pady=2, command=self._on_save_chat,
                activebackground=BORDER, activeforeground=FG,
            ).pack(side=tk.LEFT, padx=(8, 0))
        if self._diarize_available:
            self._diarize_var = tk.BooleanVar(value=self._diarize_initial)
            tk.Checkbutton(inner, text="Diarize", variable=self._diarize_var, font=("Segoe UI", 9), fg=FG_DIM, bg=BG_PANEL,
                          selectcolor=BG_INPUT, activebackground=BG_PANEL, activeforeground=FG, command=self._on_diarize_change).pack(side=tk.LEFT, padx=(12, 0))
        self._stop_btn = tk.Button(inner, text="Stop", font=("Segoe UI", 9), bg=BORDER, fg=FG, relief=tk.FLAT, padx=12, pady=2,
                                   command=self._on_stop_click, activebackground=FG_DIM, activeforeground=FG)
        self._stop_btn.pack(side=tk.LEFT, padx=(12, 0))
        tk.Button(inner, text="Close", font=("Segoe UI", 9), bg="#991b1b", fg=FG, relief=tk.FLAT, padx=12, pady=2,
                  command=self._on_close, activebackground="#b91c1c", activeforeground=FG).pack(side=tk.RIGHT)

        # --- Main content: interview = captions+context (left) | generate+answer (right) ---
        if self._interview_mode:
            outer = tk.PanedWindow(self.frame, orient=tk.HORIZONTAL, bg=BG, sashwidth=8)
            left_paned = tk.PanedWindow(outer, orient=tk.VERTICAL, bg=BG, sashwidth=8)
            dialog_parent = left_paned
        else:
            paned = tk.PanedWindow(self.frame, orient=tk.VERTICAL, bg=BG, sashwidth=8)
            dialog_parent = paned

        # --- Panel 1: Live Caption ---
        dialog_panel, dialog_content = _make_panel(dialog_parent, "Live Caption", "●")
        legend = tk.Frame(dialog_content, bg=BG_PANEL)
        legend.pack(fill=tk.X, pady=(0, 4))
        tk.Label(legend, text="Confirmed", font=("Segoe UI", 8), fg=SUCCESS, bg=BG_PANEL).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(legend, text="Live", font=("Segoe UI", 8), fg=FG_DIM, bg=BG_PANEL).pack(side=tk.LEFT)
        f = tkfont.Font(family="Segoe UI", size=self.font_size, weight="normal")
        text_frame = tk.Frame(dialog_content, bg=BG_PANEL)
        self.text = tk.Text(
            text_frame, font=f, fg=FG, bg=BG_INPUT,
            wrap=tk.WORD, insertbackground=FG, selectbackground=ACCENT, selectforeground=FG,
            relief=tk.FLAT, padx=12, pady=10, height=10,
        )
        sb1 = tk.Scrollbar(text_frame, command=self.text.yview, bg=BG_PANEL, troughcolor=BORDER)
        self.text.configure(yscrollcommand=sb1.set)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb1.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.insert(tk.END, "Listening...")
        self._final_end = self.text.index(tk.END)
        self._current_real_end = self._final_end
        text_frame.pack(fill=tk.BOTH, expand=True)
        dialog_content.pack(fill=tk.BOTH, expand=True)
        dialog_parent.add(dialog_panel, minsize=140)

        if self._interview_mode:
            # --- Left column: User Context ---
            ctx_panel, ctx_content = _make_panel(left_paned, "User Context (resume, job description, talking points)", "")
            self.context_text, ctx_inner = _make_text(ctx_content, height=5)
            ctx_inner.pack(fill=tk.BOTH, expand=True)
            left_paned.add(ctx_panel, minsize=80)

            outer.add(left_paned, minsize=400)

            # --- Right column: Generate + answer ---
            right_col = tk.Frame(outer, bg=BG, padx=2)
            btn_frame = tk.Frame(right_col, bg=BG)
            self._gen_btn = tk.Button(
                btn_frame, text="Generate Answer",
                font=("Segoe UI", 12, "bold"), bg=ACCENT, fg=FG,
                relief=tk.FLAT, padx=20, pady=10,
                command=self._on_generate,
                activebackground=ACCENT_HOVER, activeforeground=FG,
            )
            self._gen_btn.pack(fill=tk.X)
            btn_frame.pack(fill=tk.X, pady=(0, 6))

            tk.Label(
                right_col,
                text="Click answer text to copy to clipboard (e.g. paste in Notepad)",
                font=("Segoe UI", 8),
                fg=FG_DIM,
                bg=BG,
                wraplength=300,
                justify=tk.LEFT,
            ).pack(anchor=tk.W, pady=(0, 4))

            ans_panel, ans_content = _make_panel(right_col, "Generated Answer", "★")
            self.answer_text, ans_inner = _make_text(ans_content, height=12, readonly=False)
            ans_inner.pack(fill=tk.BOTH, expand=True)
            ans_panel.pack(fill=tk.BOTH, expand=True)
            self.answer_text.bind("<Button-1>", self._on_answer_click, add="+")

            outer.add(right_col, minsize=300)
            outer.pack(fill=tk.BOTH, expand=True)
        else:
            paned.pack(fill=tk.BOTH, expand=True)

        # --- Status bar ---
        status_bar = tk.Frame(self.frame, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
        status_bar.pack(fill=tk.X, pady=(8, 0))
        _status0 = (
            "Ready • click Generated Answer to copy • Esc to close"
            if self._interview_mode
            else "Ready • Ctrl+C to copy • Esc to close"
        )
        self._status_text = tk.Label(status_bar, text=_status0, font=("Segoe UI", 8), fg=FG_DIM, bg=BG_PANEL)
        self._status_text.pack(side=tk.LEFT, padx=10, pady=4)

        def block_caption_edit(e):
            if (e.state & 0x4) and e.keysym.lower() == "c":
                return
            return "break"
        self.text.tag_configure("preview", foreground=FG_DIM)
        self.text.tag_configure("final", background="#1e3a2f", foreground=FG)
        self.text.bind("<Key>", block_caption_edit)
        self.root.bind("<Escape>", lambda e: self._on_close())
        self.root.bind("<Button-3>", lambda e: self._show_menu(e))
        for w in (self.root, self.text):
            w.bind("<Control-c>", lambda e: self._copy_from(self.text))

    def _copy_from(self, widget):
        try:
            sel = widget.get(tk.SEL_FIRST, tk.SEL_LAST)
            if sel:
                self.root.clipboard_clear()
                self.root.clipboard_append(sel)
                self._set_status("Copied to clipboard")
        except tk.TclError:
            pass

    def _set_status(self, msg: str):
        if hasattr(self, "_status_text"):
            self._status_text.config(text=msg)

    def set_status(self, msg: str):
        """Public API for main to update status bar."""
        self._set_status(msg)

    def _copy_all(self):
        try:
            content = self.text.get(1.0, tk.END).strip()
            if content and not content.startswith("Listening") and not content.startswith("Stopped"):
                self.root.clipboard_clear()
                self.root.clipboard_append(content)
                self._set_status("Copied full transcript")
        except tk.TclError:
            pass

    def _on_answer_click(self, event):
        """Copy full answer on click so it can be pasted elsewhere (Notepad, etc.)."""
        if not self._interview_mode or not hasattr(self, "answer_text"):
            return
        w = event.widget
        if w is not self.answer_text:
            return
        try:
            t = self.answer_text.get(1.0, tk.END).strip()
            if t:
                self.root.clipboard_clear()
                self.root.clipboard_append(t)
                self._set_status("Copied answer — paste anywhere (e.g. Notepad)")
        except tk.TclError:
            pass

    def _on_generate(self):
        if self._generate_cb:
            self._generate_cb()

    def on_generate(self, callback):
        self._generate_cb = callback

    def _on_save_chat(self):
        if self._save_chat_cb:
            self._save_chat_cb()

    def on_save_chat(self, callback):
        """Optional: save dialogue history (wired from main)."""
        self._save_chat_cb = callback

    def _on_mode_change(self, value):
        if self._mode_change_cb:
            self._mode_change_cb(value)

    def _on_stop_click(self):
        if self._stop_cb:
            self._stop_cb()

    def _on_diarize_change(self):
        if self._diarize_change_cb:
            self._diarize_change_cb(self._diarize_var.get())

    def _on_transcriber_settings_changed(self):
        pass  # No-op; Apply button triggers the callback

    def _apply_transcriber_settings(self):
        if self._transcriber_apply_cb:
            self._transcriber_apply_cb()

    def get_device(self) -> str:
        return (self._device_var.get() or "auto").lower()

    def get_model(self) -> str:
        return self._model_var.get() or "small.en"

    def on_transcriber_apply(self, callback):
        self._transcriber_apply_cb = callback

    def set_capturing(self, capturing: bool):
        self._capturing = capturing
        self._stop_btn.config(text="Start" if not capturing else "Stop")
        self._update_status_indicator(capturing)

    def _update_status_indicator(self, capturing: bool):
        if not hasattr(self, "_status_dot"):
            return
        self._status_dot.delete("all")
        color = LIVE if capturing else FG_DIM
        self._status_dot.create_oval(1, 1, 9, 9, fill=color, outline="")
        if hasattr(self, "_status_label"):
            self._status_label.config(text="LIVE" if capturing else "STOPPED", fg=color)

    def is_capturing(self) -> bool:
        return self._capturing

    def on_mode_change(self, callback):
        self._mode_change_cb = callback

    def clear_history(self):
        self._clear_preview()
        if self._clear_cb:
            self._clear_cb()
        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END, "Listening...")
        self._final_end = self.text.index(tk.END)
        self._current_real_end = self._final_end
        self._last_speaker = None
        if self._interview_mode and hasattr(self, "answer_text"):
            self.answer_text.config(state=tk.NORMAL)
            self.answer_text.delete(1.0, tk.END)
            self.answer_text.config(state=tk.NORMAL)
        self._set_status("Chat cleared • Ready")

    def get_mode(self):
        return self._mode_var.get()

    def set_mode(self, mode: str):
        self._mode_var.set(mode)

    def _make_draggable(self):
        def on_press(e):
            self._drag_x, self._drag_y = e.x, e.y
        def on_drag(e):
            x = self.root.winfo_x() + e.x - self._drag_x
            y = self.root.winfo_y() + e.y - self._drag_y
            self.root.geometry(f"+{x}+{y}")
        self.frame.bind("<Button-1>", on_press)
        self.frame.bind("<B1-Motion>", on_drag)
        self.text.bind("<Button-1>", on_press)
        self.text.bind("<B1-Motion>", on_drag)
        self.root.bind("<Button-1>", on_press)
        self.root.bind("<B1-Motion>", on_drag)

    def _on_close(self):
        if hasattr(self, "_close_cb") and self._close_cb:
            self._close_cb()

    def _show_menu(self, event):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Copy selection (Ctrl+C)", command=lambda: self._copy_from(self.text))
        menu.add_command(label="Copy all", command=lambda: self._copy_all())
        menu.add_command(label="Clear history", command=self.clear_history)
        menu.add_separator()
        menu.add_command(label="Close (Esc)", command=self._on_close)
        menu.tk_popup(event.x_root, event.y_root)

    def on_close(self, callback):
        self._close_cb = callback

    def on_stop(self, callback):
        self._stop_cb = callback

    def on_clear(self, callback):
        """Called when user clicks Clear (before clearing UI)."""
        self._clear_cb = callback

    def on_diarize_change(self, callback):
        self._diarize_change_cb = callback

    def get_diarize(self) -> bool:
        return self._diarize_var.get() if self._diarize_available else False

    def get_user_context(self) -> str:
        """Get preloaded context text (resume, job description, etc.)."""
        if self._interview_mode and hasattr(self, "context_text"):
            return self.context_text.get(1.0, tk.END).strip()
        return ""

    def set_generated_answer(self, text: str):
        """Display the generated answer in the answer panel."""
        if self._interview_mode and hasattr(self, "answer_text"):
            self.answer_text.config(state=tk.NORMAL)
            self.answer_text.delete(1.0, tk.END)
            self.answer_text.insert(tk.END, text or "")
            self.answer_text.config(state=tk.NORMAL)
            if (
                hasattr(self, "_status_text")
                and text
                and not getattr(self, "_answer_streaming", False)
            ):
                self._set_status("Answer generated")

    def set_generate_loading(self, loading: bool):
        """Show loading state on Generate button."""
        self._answer_streaming = loading
        if hasattr(self, "_gen_btn"):
            self._gen_btn.config(
                state=tk.DISABLED if loading else tk.NORMAL,
                text="Generating..." if loading else "Generate Answer",
            )

    def set_text(self, text: str):
        self._clear_preview()
        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END, text or "Listening...")
        self._final_end = self.text.index(tk.END)
        self._current_real_end = self._final_end
        self.text.see(tk.END)

    def _show_preview(self, text: str):
        """Legacy: replace preview in place. Prefer set_temp for three-layer flow."""
        self.set_temp(text)

    def _strip_overlap_words(self, committed_words: list, new_words: list) -> list:
        """If end of committed matches start of new, return only remainder."""
        def _w_norm(w): return w.lower().rstrip(".,?!;:\"'")
        for k in range(min(len(committed_words), len(new_words)), 0, -1):
            if all(_w_norm(c) == _w_norm(n) for c, n in zip(committed_words[-k:], new_words[:k])):
                return new_words[k:]
        return new_words

    def append_final(self, text: str, speaker: str = None):
        """Append to final region with overlap dedup. Replaces current_real+temp with deduped segment."""
        text_stripped = text.strip()
        if not text_stripped:
            self.clear_segment()
            return
        content = self.text.get(1.0, self._final_end).strip()
        if content.startswith("Listening") or content.startswith("Stopped"):
            try:
                self.text.delete(1.0, tk.END)
            except tk.TclError:
                pass
            self._final_end = "1.0"
            self._current_real_end = "1.0"
            self._last_speaker = None
        try:
            self.text.delete(self._final_end, tk.END)
        except tk.TclError:
            pass
        if speaker:
            label = speaker.replace("_", " ").title()
            line = f"{label} : {text_stripped}"
            if self._last_speaker == speaker and content and not content.startswith("Listening") and not content.startswith("Stopped"):
                prev_text = content.split(" : ", 1)[-1].strip() if " : " in content else content
                cw, tw = prev_text.split(), text_stripped.split()
                to_add_words = self._strip_overlap_words(cw, tw)
                to_add = " ".join(to_add_words) if to_add_words else ""
                if to_add:
                    sep = "" if not content.endswith(" ") and not content.endswith("\n") else " "
                    self.text.insert(self._final_end, sep + to_add, "final")
            else:
                prefix = "" if not content or content.startswith("Listening") or content.startswith("Stopped") else "\n"
                self.text.insert(self._final_end, prefix + line, "final")
            self._last_speaker = speaker
        else:
            cw, tw = content.split(), text_stripped.split()
            to_add_words = self._strip_overlap_words(cw, tw)
            to_add = " ".join(to_add_words) if to_add_words else ""
            if to_add:
                sep = "" if not content or content.endswith(" ") or content.endswith("\n") else " "
                if not content or content.startswith("Listening") or content.startswith("Stopped"):
                    sep = ""
                self.text.insert(self._final_end, (sep + to_add) if sep or to_add else to_add, "final")
            self._last_speaker = None
        self._final_end = self.text.index("insert")
        self._current_real_end = self._final_end
        self.text.see(tk.END)

    def append_current_real(self, delta: str):
        """Append stable delta to current real region only."""
        delta_stripped = delta.strip()
        if not delta_stripped:
            return
        content = self.text.get(self._final_end, self._current_real_end).strip()
        if content and not content.endswith(" ") and not content.endswith("\n"):
            self.text.insert(self._current_real_end, " " + delta_stripped, "final")
        else:
            self.text.insert(self._current_real_end, delta_stripped, "final")
        self._current_real_end = self.text.index("insert")
        self.text.see(tk.END)

    def set_temp(self, text: str):
        """Replace temp region entirely — never append. Updates each call."""
        try:
            self.text.delete(self._current_real_end, tk.END)
        except tk.TclError:
            pass
        if not text or not text.strip():
            self.text.see(tk.END)
            return
        to_insert = text.strip()
        existing = self.text.get(1.0, self._current_real_end).strip()
        sep = " " if existing and not existing.endswith(" ") and not existing.endswith("\n") and not to_insert.startswith(" ") else ""
        self.text.insert(self._current_real_end, sep + to_insert, "preview")
        self.text.see(tk.END)

    def clear_segment(self):
        """Clear current real and temp regions. Call after finalizing."""
        try:
            self.text.delete(self._current_real_end, tk.END)
        except tk.TclError:
            pass
        self._current_real_end = self._final_end
        self.text.see(tk.END)

    def clear_preview(self):
        """Clear the temporary preview text (call when segment finalizes)."""
        self._clear_preview()

    def _clear_preview(self):
        if self._preview_mark is not None:
            try:
                self.text.delete("preview_start", tk.END)
            except tk.TclError:
                pass
            self._preview_mark = None

    def append_text(self, text: str, speaker: str = None, is_preview: bool = False):
        """Final = highlighted. Preview = gray, replaced each update. Commit clears preview first."""
        if is_preview:
            self._show_preview(text)
            return
        self._clear_preview()  # Clear preview so committed text replaces it (preview → highlight)
        content = self.text.get(1.0, tk.END).strip()
        if content.startswith("Listening") or content.startswith("Stopped"):
            self.text.delete(1.0, tk.END)
            self._last_speaker = None
        text_stripped = text.strip()
        if speaker:
            label = speaker.replace("_", " ").title()
            line = f"{label} : {text_stripped}"
            if self._last_speaker == speaker and content and not content.startswith("Listening") and not content.startswith("Stopped"):
                cw, tw = content.split(), text_stripped.split()
                def _w_norm(w): return w.lower().rstrip(".,?!;:\"'")
                overlap = next((k for k in range(min(len(cw), len(tw)), 0, -1) if all(_w_norm(c) == _w_norm(t) for c, t in zip(cw[-k:], tw[:k]))), 0)
                to_add = " ".join(tw[overlap:]) if overlap else text_stripped
                if to_add:
                    self.text.insert(tk.END, " " + to_add, "final")
            else:
                prefix = "" if not content or content.startswith("Listening") or content.startswith("Stopped") else "\n"
                self.text.insert(tk.END, prefix + line, "final")
            self._last_speaker = speaker
        else:
            prev_ends_sentence = content and content[-1:] in ".?!。？！"
            prev_ends_comma = content and content.rstrip()[-1:] in ",;:"
            starts_continuation = (
                text_stripped and (text_stripped[0].islower() or text_stripped.lower().startswith(
                    ("and ", "or ", "but ", "the ", "to ", "of ", "in ", "on ", "at ", "so ", "because ", "also ", "your ", "our ", "it ", "is ", "yeah ", "yes ", "well ")
                ))
            )
            ends_with_ellipsis = text_stripped.endswith(("...", "…"))
            should_merge = (
                content and not content.startswith("Listening") and not content.startswith("Stopped")
                and text_stripped
                and (not prev_ends_sentence or starts_continuation or prev_ends_comma or ends_with_ellipsis)
                and (len(text_stripped) <= 100 or starts_continuation or ends_with_ellipsis)
                and not text_stripped.lstrip().startswith((".", "?", "!"))
            )
            if should_merge:
                cw, tw = content.split(), text_stripped.split()
                def _w_norm(w): return w.lower().rstrip(".,?!;:\"'")
                overlap = next((k for k in range(min(len(cw), len(tw)), 0, -1) if all(_w_norm(c) == _w_norm(t) for c, t in zip(cw[-k:], tw[:k]))), 0)
                to_add = " ".join(tw[overlap:]) if overlap else text_stripped
                if to_add:
                    self.text.insert(tk.END, " " + to_add, "final")
            else:
                prefix = "" if not content or content.startswith("Listening") or content.startswith("Stopped") else "\n"
                self.text.insert(tk.END, prefix + text, "final")
            self._last_speaker = None
        self.text.see(tk.END)

    def run(self):
        self.root.geometry(f"{self.width}x{self.height}+80+40")
        self.root.mainloop()

    def close(self):
        self._close_cb = None
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
