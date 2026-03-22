"""
Interview Answering Assistant UI.
Panels: Live Caption/Dialogue History, User Context, Generated Answer.
"""

import tkinter as tk
from tkinter import font as tkfont

# Style constants
BG = "#1a1a1a"
FG = "#ffffff"
ACCENT = "#4a9eff"


def _make_label(parent, text: str, bg=BG, fg="#aaa"):
    """Create a panel section label."""
    return tk.Label(parent, text=text, font=("Segoe UI", 9), fg=fg, bg=bg)


def _make_text(parent, height: int, readonly: bool = False, **kw):
    """Create a Text widget with scrollbar."""
    frame = tk.Frame(parent, bg=BG)
    t = tk.Text(
        frame,
        font=("Segoe UI", 11),
        fg=FG,
        bg="#252525",
        wrap=tk.WORD,
        insertbackground=FG,
        selectbackground="#404040",
        selectforeground=FG,
        relief=tk.FLAT,
        padx=6,
        pady=4,
        height=height,
        **kw,
    )
    sb = tk.Scrollbar(frame, command=t.yview, bg=BG, troughcolor=BG)
    t.configure(yscrollcommand=sb.set)
    t.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    if readonly:
        t.config(state=tk.DISABLED)
    return t, frame


class CaptionWindow:
    def __init__(
        self,
        width=920,
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
        self.root.title("Interview Answering Assistant")
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
        self._clear_cb = None
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
        self.frame = tk.Frame(self.root, bg=self.bg, padx=10, pady=6)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # --- Toolbar ---
        toolbar = tk.Frame(self.frame, bg=self.bg)
        toolbar.pack(fill=tk.X, pady=(0, 6))
        tk.Label(toolbar, text="Capture:", font=("Segoe UI", 10), fg="#aaa", bg=self.bg).pack(side=tk.LEFT, padx=(0, 6))
        default_mode = "system" if self._system_available else "mic"
        self._mode_var = tk.StringVar(value=default_mode)
        modes = ["system", "both"] if self._system_available else []
        modes.append("mic")
        self._mode_menu = tk.OptionMenu(toolbar, self._mode_var, *modes, command=self._on_mode_change)
        self._mode_menu.config(font=("Segoe UI", 10), bg="#2a2a2a", fg=self.fg, highlightthickness=0, anchor="w")
        self._mode_menu.pack(side=tk.LEFT)
        tk.Button(toolbar, text="Clear", font=("Segoe UI", 9), bg="#333", fg=self.fg, relief=tk.FLAT, padx=8, command=self.clear_history).pack(side=tk.LEFT, padx=(12, 0))
        if self._diarize_available:
            self._diarize_var = tk.BooleanVar(value=self._diarize_initial)
            tk.Checkbutton(toolbar, text="Diarize", variable=self._diarize_var, font=("Segoe UI", 9), fg="#aaa", bg=self.bg, selectcolor="#333", activebackground=self.bg, activeforeground=self.fg, command=self._on_diarize_change).pack(side=tk.LEFT, padx=(12, 0))
        self._stop_btn = tk.Button(toolbar, text="Stop", font=("Segoe UI", 9), bg="#444", fg=self.fg, relief=tk.FLAT, padx=10, command=self._on_stop_click)
        self._stop_btn.pack(side=tk.LEFT, padx=(12, 0))
        tk.Button(toolbar, text="Close", font=("Segoe UI", 9), bg="#c44", fg=self.fg, relief=tk.FLAT, padx=10, command=self._on_close).pack(side=tk.RIGHT)

        # --- Main content: PanedWindow ---
        paned = tk.PanedWindow(self.frame, orient=tk.VERTICAL, bg=self.bg, sashwidth=6)

        # --- Panel 1: Live Caption / Dialogue History ---
        dialog_frame = tk.Frame(paned, bg=self.bg)
        _make_label(dialog_frame, "Live Caption / Dialogue History").pack(anchor=tk.W)
        f = tkfont.Font(family="Segoe UI", size=self.font_size, weight="normal")
        text_frame = tk.Frame(dialog_frame, bg=self.bg)
        self.text = tk.Text(
            text_frame, font=f, fg=self.fg, bg=self.bg,
            wrap=tk.WORD, insertbackground=self.fg, selectbackground="#404040", selectforeground=self.fg,
            relief=tk.FLAT, padx=4, pady=4, height=10,
        )
        sb1 = tk.Scrollbar(text_frame, command=self.text.yview, bg=self.bg, troughcolor=self.bg)
        self.text.configure(yscrollcommand=sb1.set)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb1.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.insert(tk.END, "Listening...")
        self._final_end = self.text.index(tk.END)
        self._current_real_end = self._final_end
        text_frame.pack(fill=tk.BOTH, expand=True)
        dialog_frame.pack(fill=tk.BOTH, expand=True)
        paned.add(dialog_frame, minsize=120)

        # --- Panel 2: User Context ---
        if self._interview_mode:
            ctx_frame = tk.Frame(paned, bg=self.bg)
            _make_label(ctx_frame, "User Context (resume, job description, talking points)").pack(anchor=tk.W)
            self.context_text, ctx_inner = _make_text(ctx_frame, height=6)
            ctx_inner.pack(fill=tk.BOTH, expand=True)
            paned.add(ctx_frame, minsize=80)

            # --- Generate Answer button ---
            btn_frame = tk.Frame(paned, bg=self.bg)
            self._gen_btn = tk.Button(
                btn_frame, text="Generate Answer",
                font=("Segoe UI", 11), bg=ACCENT, fg=FG,
                relief=tk.FLAT, padx=20, pady=8,
                command=self._on_generate,
            )
            self._gen_btn.pack(pady=4)
            paned.add(btn_frame, minsize=50)

            # --- Panel 3: Generated Answer ---
            ans_frame = tk.Frame(paned, bg=self.bg)
            _make_label(ans_frame, "Generated Answer").pack(anchor=tk.W)
            self.answer_text, ans_inner = _make_text(ans_frame, height=8, readonly=False)
            ans_inner.pack(fill=tk.BOTH, expand=True)
            paned.add(ans_frame, minsize=100)

        paned.pack(fill=tk.BOTH, expand=True)

        def block_caption_edit(e):
            if (e.state & 0x4) and e.keysym.lower() == "c":
                return
            return "break"
        # Temp/preview = dim gray; real/final = highlighted (subtle background)
        self.text.tag_configure("preview", foreground="#888888")
        self.text.tag_configure("final", background="#2a3540", foreground=self.fg)  # subtle highlight for confirmed text
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
        except tk.TclError:
            pass

    def _on_generate(self):
        if self._generate_cb:
            self._generate_cb()

    def on_generate(self, callback):
        self._generate_cb = callback

    def _on_mode_change(self, value):
        if self._mode_change_cb:
            self._mode_change_cb(value)

    def _on_stop_click(self):
        if self._stop_cb:
            self._stop_cb()

    def _on_diarize_change(self):
        if self._diarize_change_cb:
            self._diarize_change_cb(self._diarize_var.get())

    def set_capturing(self, capturing: bool):
        self._capturing = capturing
        self._stop_btn.config(text="Start" if not capturing else "Stop")

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
        menu.add_command(label="Copy (Ctrl+C)", command=lambda: self._copy_from(self.text))
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

    def set_generate_loading(self, loading: bool):
        """Show loading state on Generate button."""
        if hasattr(self, "_gen_btn"):
            self._gen_btn.config(state=tk.DISABLED if loading else tk.NORMAL, text="Generating..." if loading else "Generate Answer")

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
