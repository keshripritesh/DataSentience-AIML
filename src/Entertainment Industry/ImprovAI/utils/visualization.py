"""
Visualization utilities for ImprovAI.
Provides piano roll, waveform, and other musical visualizations.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

from core.encoders.music_encoder import Note, MusicalSequence


def create_piano_roll(notes: List[Note], title: str = "Piano Roll") -> go.Figure:
    """
    Create a piano roll visualization of musical notes.
    
    Args:
        notes: List of Note objects
        title: Title for the plot
        
    Returns:
        Plotly figure object
    """
    if not notes:
        # Create empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No notes to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=title,
            xaxis_title="Time (seconds)",
            yaxis_title="MIDI Pitch",
            height=400
        )
        return fig
    
    # Extract data
    pitches = [note.pitch for note in notes]
    start_times = [note.start_time for note in notes]
    durations = [note.duration for note in notes]
    velocities = [note.velocity for note in notes]
    
    # Create figure
    fig = go.Figure()
    
    # Add note rectangles
    for i, (pitch, start, duration, velocity) in enumerate(zip(pitches, start_times, durations, velocities)):
        # Normalize velocity for color intensity
        color_intensity = velocity / 127.0
        
        fig.add_shape(
            type="rect",
            x0=start,
            x1=start + duration,
            y0=pitch - 0.4,
            y1=pitch + 0.4,
            fillcolor=f"rgba(30, 144, 255, {color_intensity})",
            line=dict(color="rgba(30, 144, 255, 0.8)", width=1),
            layer="below"
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="MIDI Pitch",
        height=400,
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(
            gridcolor="lightgray",
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            gridcolor="lightgray",
            showgrid=True,
            zeroline=False,
            tickmode="array",
            tickvals=list(range(min(pitches), max(pitches) + 1, 12)),
            ticktext=[f"C{octave}" for octave in range(min(pitches)//12, max(pitches)//12 + 1)]
        )
    )
    
    return fig


def create_waveform_plot(audio_data: np.ndarray, sample_rate: int, title: str = "Waveform") -> go.Figure:
    """
    Create a waveform visualization of audio data.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate in Hz
        title: Title for the plot
        
    Returns:
        Plotly figure object
    """
    # Create time axis
    time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=audio_data,
        mode='lines',
        line=dict(color='blue', width=1),
        name='Waveform'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=300,
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(gridcolor="lightgray", showgrid=True),
        yaxis=dict(gridcolor="lightgray", showgrid=True)
    )
    
    return fig


def create_spectrogram_plot(audio_data: np.ndarray, sample_rate: int, title: str = "Spectrogram") -> go.Figure:
    """
    Create a spectrogram visualization of audio data.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate in Hz
        title: Title for the plot
        
    Returns:
        Plotly figure object
    """
    # Compute spectrogram
    from scipy import signal
    
    frequencies, times, Sxx = signal.spectrogram(
        audio_data, 
        sample_rate, 
        nperseg=1024, 
        noverlap=512
    )
    
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    fig = go.Figure(data=go.Heatmap(
        z=Sxx_db,
        x=times,
        y=frequencies,
        colorscale='Viridis',
        colorbar=dict(title="Power (dB)")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
        height=400,
        yaxis=dict(type="log")
    )
    
    return fig


def create_harmonic_analysis_plot(notes: List[Note], title: str = "Harmonic Analysis") -> go.Figure:
    """
    Create a harmonic analysis visualization.
    
    Args:
        notes: List of Note objects
        title: Title for the plot
        
    Returns:
        Plotly figure object
    """
    if not notes:
        return create_empty_plot("No notes for harmonic analysis")
    
    # Extract pitch classes
    pitch_classes = [note.pitch % 12 for note in notes]
    
    # Count occurrences
    pc_counts = np.zeros(12)
    for pc in pitch_classes:
        pc_counts[pc] += 1
    
    # Create bar chart
    pc_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    fig = go.Figure(data=go.Bar(
        x=pc_names,
        y=pc_counts,
        marker_color='lightblue',
        marker_line_color='blue',
        marker_line_width=1
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Pitch Class",
        yaxis_title="Frequency",
        height=300,
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(gridcolor="lightgray", showgrid=True),
        yaxis=dict(gridcolor="lightgray", showgrid=True)
    )
    
    return fig


def create_rhythm_analysis_plot(notes: List[Note], title: str = "Rhythm Analysis") -> go.Figure:
    """
    Create a rhythm analysis visualization.
    
    Args:
        notes: List of Note objects
        title: Title for the plot
        
    Returns:
        Plotly figure object
    """
    if not notes:
        return create_empty_plot("No notes for rhythm analysis")
    
    # Extract onset times
    onset_times = sorted(list(set([note.start_time for note in notes])))
    
    if len(onset_times) < 2:
        return create_empty_plot("Insufficient notes for rhythm analysis")
    
    # Calculate inter-onset intervals
    intervals = np.diff(onset_times)
    
    # Create histogram
    fig = go.Figure(data=go.Histogram(
        x=intervals,
        nbinsx=20,
        marker_color='lightgreen',
        marker_line_color='green',
        marker_line_width=1
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Inter-onset Interval (seconds)",
        yaxis_title="Frequency",
        height=300,
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(gridcolor="lightgray", showgrid=True),
        yaxis=dict(gridcolor="lightgray", showgrid=True)
    )
    
    return fig


def create_melodic_contour_plot(notes: List[Note], title: str = "Melodic Contour") -> go.Figure:
    """
    Create a melodic contour visualization.
    
    Args:
        notes: List of Note objects
        title: Title for the plot
        
    Returns:
        Plotly figure object
    """
    if not notes:
        return create_empty_plot("No notes for melodic contour")
    
    # Sort notes by start time
    sorted_notes = sorted(notes, key=lambda x: x.start_time)
    
    # Extract pitches and times
    pitches = [note.pitch for note in sorted_notes]
    times = [note.start_time for note in sorted_notes]
    
    fig = go.Figure()
    
    # Add melodic line
    fig.add_trace(go.Scatter(
        x=times,
        y=pitches,
        mode='lines+markers',
        line=dict(color='red', width=2),
        marker=dict(size=6, color='red'),
        name='Melodic Contour'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="MIDI Pitch",
        height=300,
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(gridcolor="lightgray", showgrid=True),
        yaxis=dict(gridcolor="lightgray", showgrid=True)
    )
    
    return fig


def create_velocity_analysis_plot(notes: List[Note], title: str = "Velocity Analysis") -> go.Figure:
    """
    Create a velocity analysis visualization.
    
    Args:
        notes: List of Note objects
        title: Title for the plot
        
    Returns:
        Plotly figure object
    """
    if not notes:
        return create_empty_plot("No notes for velocity analysis")
    
    # Extract velocities and times
    velocities = [note.velocity for note in notes]
    times = [note.start_time for note in notes]
    
    fig = go.Figure()
    
    # Add velocity line
    fig.add_trace(go.Scatter(
        x=times,
        y=velocities,
        mode='lines+markers',
        line=dict(color='purple', width=2),
        marker=dict(size=6, color='purple'),
        name='Velocity'
    ))
    
    # Add mean velocity line
    mean_velocity = np.mean(velocities)
    fig.add_hline(
        y=mean_velocity,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Mean: {mean_velocity:.1f}"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Velocity",
        height=300,
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(gridcolor="lightgray", showgrid=True),
        yaxis=dict(gridcolor="lightgray", showgrid=True, range=[0, 127])
    )
    
    return fig


def create_quality_metrics_plot(metrics: Dict[str, float], title: str = "Quality Metrics") -> go.Figure:
    """
    Create a quality metrics visualization.
    
    Args:
        metrics: Dictionary of quality metrics
        title: Title for the plot
        
    Returns:
        Plotly figure object
    """
    if not metrics:
        return create_empty_plot("No quality metrics available")
    
    # Filter out overall_quality if present
    plot_metrics = {k: v for k, v in metrics.items() if k != 'overall_quality'}
    
    if not plot_metrics:
        return create_empty_plot("No individual metrics available")
    
    # Create radar chart
    categories = list(plot_metrics.keys())
    values = list(plot_metrics.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Quality Metrics',
        line_color='blue'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title=title,
        height=400
    )
    
    return fig


def create_comparison_plot(sequences: List[MusicalSequence], 
                          labels: List[str], 
                          title: str = "Sequence Comparison") -> go.Figure:
    """
    Create a comparison plot of multiple musical sequences.
    
    Args:
        sequences: List of MusicalSequence objects
        labels: List of labels for each sequence
        title: Title for the plot
        
    Returns:
        Plotly figure object
    """
    if not sequences or len(sequences) != len(labels):
        return create_empty_plot("Invalid sequences or labels")
    
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (sequence, label) in enumerate(zip(sequences, labels)):
        if not sequence.notes:
            continue
        
        # Extract data
        pitches = [note.pitch for note in sequence.notes]
        times = [note.start_time for note in sequence.notes]
        
        # Add sequence line
        fig.add_trace(go.Scatter(
            x=times,
            y=pitches,
            mode='lines+markers',
            name=label,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="MIDI Pitch",
        height=400,
        plot_bgcolor="white",
        xaxis=dict(gridcolor="lightgray", showgrid=True),
        yaxis=dict(gridcolor="lightgray", showgrid=True)
    )
    
    return fig


def create_empty_plot(message: str = "No data to display") -> go.Figure:
    """
    Create an empty plot with a message.
    
    Args:
        message: Message to display
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        height=300
    )
    return fig


def create_matplotlib_piano_roll(notes: List[Note], title: str = "Piano Roll") -> plt.Figure:
    """
    Create a matplotlib piano roll visualization.
    
    Args:
        notes: List of Note objects
        title: Title for the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if not notes:
        ax.text(0.5, 0.5, "No notes to display", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    # Extract data
    pitches = [note.pitch for note in notes]
    start_times = [note.start_time for note in notes]
    durations = [note.duration for note in notes]
    velocities = [note.velocity for note in notes]
    
    # Create rectangles for each note
    for pitch, start, duration, velocity in zip(pitches, start_times, durations, velocities):
        # Normalize velocity for color intensity
        color_intensity = velocity / 127.0
        
        rect = patches.Rectangle(
            (start, pitch - 0.4),
            duration,
            0.8,
            facecolor=(0.2, 0.6, 1.0, color_intensity),
            edgecolor=(0.2, 0.6, 1.0, 0.8),
            linewidth=1
        )
        ax.add_patch(rect)
    
    # Set axis properties
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("MIDI Pitch")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis ticks for octaves
    if pitches:
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        octave_ticks = list(range(min_pitch - (min_pitch % 12), max_pitch + 12, 12))
        octave_labels = [f"C{octave//12}" for octave in octave_ticks]
        ax.set_yticks(octave_ticks)
        ax.set_yticklabels(octave_labels)
    
    return fig


def save_visualization(fig, filepath: str, format: str = "png", dpi: int = 300):
    """
    Save a plotly figure to file.
    
    Args:
        fig: Plotly figure object
        filepath: Output file path
        format: Image format (png, jpg, svg, pdf)
        dpi: DPI for raster formats
    """
    try:
        fig.write_image(filepath, format=format, width=800, height=600, scale=2)
        print(f"Visualization saved to {filepath}")
    except Exception as e:
        print(f"Error saving visualization: {e}")


def create_interactive_dashboard(notes: List[Note]) -> go.Figure:
    """
    Create an interactive dashboard with multiple visualizations.
    
    Args:
        notes: List of Note objects
        
    Returns:
        Plotly figure with subplots
    """
    from plotly.subplots import make_subplots
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Piano Roll", "Melodic Contour", "Harmonic Analysis", "Rhythm Analysis"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    if not notes:
        # Add empty annotations
        for i in range(1, 3):
            for j in range(1, 3):
                fig.add_annotation(
                    text="No data",
                    xref=f"x{i}{j}", yref=f"y{i}{j}",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=16, color="gray")
                )
        return fig
    
    # Piano Roll (top left)
    piano_fig = create_piano_roll(notes, "")
    for trace in piano_fig.data:
        fig.add_trace(trace, row=1, col=1)
    
    # Melodic Contour (top right)
    contour_fig = create_melodic_contour_plot(notes, "")
    for trace in contour_fig.data:
        fig.add_trace(trace, row=1, col=2)
    
    # Harmonic Analysis (bottom left)
    harmonic_fig = create_harmonic_analysis_plot(notes, "")
    for trace in harmonic_fig.data:
        fig.add_trace(trace, row=2, col=1)
    
    # Rhythm Analysis (bottom right)
    rhythm_fig = create_rhythm_analysis_plot(notes, "")
    for trace in rhythm_fig.data:
        fig.add_trace(trace, row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title="Musical Analysis Dashboard",
        height=800,
        showlegend=False
    )
    
    return fig
