import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(layout="wide")

name_mapping = {
    "pitch": "max_pitch_semitones_above_50_hz",
    "intensity": "max_intensity",
    "backchannels": "backchannel_overlap",
}

inv_name_mapping = {val: key for key, val in name_mapping.items()}

matplotlib.rcParams["savefig.format"] = "svg"


@st.cache_data
def load_backchannels():
    # df = pd.read_csv("../data/candor/backchannels.csv")
    df = pd.read_csv("./demo_backchannels.csv")
    return df


@st.cache_data
def load_data():
    data = pd.read_csv("./demo_data.csv")
    # dfs = []
    # for context_len in range(0, 6):
    #     df = pd.read_csv(
    #         f"../output/candor/merged/gpt2/{context_len}/backbiter/candor_merged_data_gpt2_{context_len}_backbiter_explore.csv"
    #     )
    #     df["context_len"] = context_len
    #     dfs.append(df)
    # data = pd.concat(dfs)
    # data = data[
    #     [
    #         "transcript_name",
    #         "turn_id",
    #         "word",
    #         "start_time",
    #         "end_time",
    #         "max_intensity",
    #         "max_pitch_semitones_above_50_hz",
    #         "backchannel_overlap",
    #         "speaker",
    #         "surprisal",
    #         "entropy",
    #         "context_len",
    #         "post_word_pause",
    #     ]
    # ]
    # data["parity"] = data.groupby("speaker").cumcount() % 2 == 0
    return data


def make_plot(
    feature1,
    feature2,
    filtered_data,
    context_len,
    convo_name,
    start_t,
    end_t,
    backchannel_data,
):

    filtered_data = filtered_data.query("context_len == @context_len")
    filtered_data = filtered_data.query("transcript_name == @convo_name")
    filtered_data = filtered_data.query("start_time >= @start_t & end_time <= @end_t")

    backchannel_data = backchannel_data.query(
        "transcript_name == @convo_name & start >= @start_t & stop <= @end_t"
    )

    speakers = filtered_data["speaker"].unique()
    speaker_y_positions = {speaker: i for i, speaker in enumerate(speakers)}

    palette = dict(zip(speakers, ["red", "blue"]))
    parity_palette = dict(
        zip(
            [(True, 0), (False, 0), (True, 1), (False, 1)],
            ["red", "lightpink", "blue", "lightblue"],
        )
    )

    # Plotting
    fig, (ax2, ax3, ax4, ax1) = plt.subplots(
        4,
        1,
        figsize=((end_t - start_t) * 2, 6),
        height_ratios=[1, 1, 1, 1],
        sharex=True,
    )

    ax2.scatter(
        filtered_data.start_time,
        filtered_data[feature1],
        alpha=0.2,
        color=[palette.get(x) for x in filtered_data.speaker],
    )
    ax2.vlines(
        filtered_data.start_time,
        filtered_data[feature1].min() if len(filtered_data[feature1]) else 0,
        filtered_data[feature1].max() if len(filtered_data[feature1]) else 1,
        alpha=0.2,
        color="gray",
        linestyles="--",
    )

    for grp, df_sub in filtered_data.groupby(["turn_id", "speaker"]):
        ax2.plot(
            df_sub.start_time,
            df_sub[feature1],
            color=palette.get(df_sub.speaker.unique().item()),
            alpha=0.2,
        )

    ax2.set_ylim(
        filtered_data[feature1].min()
        if not np.isnan(filtered_data[feature1].min())
        else 0,
        filtered_data[feature1].max()
        if not np.isnan(filtered_data[feature1].max())
        else 1,
    )
    ax2.set_ylabel(inv_name_mapping.get(feature1, feature1))
    ax2.patch.set_visible(False)

    ax3.scatter(
        filtered_data.start_time,
        filtered_data[feature2],
        alpha=0.2,
        color=[palette.get(x) for x in filtered_data.speaker],
    )
    ax3.vlines(
        filtered_data.start_time,
        filtered_data[feature2].min() if len(filtered_data[feature2]) else 0,
        filtered_data[feature2].max() if len(filtered_data[feature2]) else 1,
        alpha=0.2,
        color="gray",
        linestyles="--",
    )

    for grp, df_sub in filtered_data.groupby(["turn_id", "speaker"]):
        ax3.plot(
            df_sub.start_time,
            df_sub[feature2],
            color=palette.get(df_sub.speaker.unique().item()),
            alpha=0.2,
        )

    ax3.set_ylim(
        filtered_data[feature2].min()
        if not np.isnan(filtered_data[feature2].min())
        else 0,
        filtered_data[feature2].max()
        if not np.isnan(filtered_data[feature2].max())
        else 1,
    )
    ax3.set_ylabel(inv_name_mapping.get(feature2, feature2))
    ax3.patch.set_visible(False)

    ax4.patch.set_visible(False)
    ax4.set_ylabel("Backchannels")
    ax4.set_ylim((-0.25, 2))
    for _, row in backchannel_data.iterrows():
        speaker_y = speaker_y_positions[row["speaker"]]
        ax4.plot(
            [row["start"], row["stop"]],
            [speaker_y, speaker_y],
            marker="|",
            ms=10,
            lw=2,
            color=palette.get(row["speaker"]),
            alpha=0.9,
            label=row["speaker"]
            if speaker_y not in plt.gca().get_legend_handles_labels()[1]
            else "",
        )

        ax4.text(
            row["start"],
            speaker_y + 0.15,
            row["utterance"],
            horizontalalignment="left",
            verticalalignment="bottom",
            fontsize=8,
            rotation=30,
            rotation_mode="anchor",
        )
    ax4.vlines(
        filtered_data.start_time, -0.25, 2.0, alpha=0.2, color="gray", linestyles="--",
    )
    ax4.set_yticks(list(speaker_y_positions.values()), list(speaker_y_positions.keys()))

    ax1.patch.set_visible(False)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Speaker")
    ax1.set_ylim((-0.25, 2))
    ax1.vlines(
        filtered_data.start_time, -0.25, 2, alpha=0.2, color="gray", linestyles="--",
    )
    for _, row in filtered_data.iterrows():
        speaker_y = speaker_y_positions[row["speaker"]]
        ax1.plot(
            [row["start_time"], row["end_time"]],
            [speaker_y, speaker_y],
            marker="|",
            ms=10,
            lw=2,
            color=parity_palette.get(
                (row["parity"], speaker_y_positions[row["speaker"]])
            ),
            alpha=0.9,
            label=row["speaker"]
            if speaker_y not in plt.gca().get_legend_handles_labels()[1]
            else "",
        )

        ax1.text(
            row["start_time"],
            speaker_y + 0.15,
            row["word"],
            horizontalalignment="left",
            verticalalignment="bottom",
            fontsize=8,
            rotation=30,
            rotation_mode="anchor",
        )
    # plt.title(f"{feature} over time")
    ax1.set_yticks(list(speaker_y_positions.values()), list(speaker_y_positions.keys()))

    plt.grid(False)
    plt.tight_layout()

    # st.pyplot(fig, format="png", dpi=180)

    svg_file = StringIO()
    fig.savefig(svg_file, format="svg")  # Save the figure to the StringIO object as SVG
    svg_file.seek(0)  # Go to the beginning of the StringIO object
    svg_data = svg_file.getvalue()  # Retrieve the SVG data

    # Close the figure to prevent it from consuming memory
    plt.close(fig)

    scrollable_container = f"""
    <div style="width: auto; overflow-x: scroll; border: 1px solid #ccc;">
        {svg_data}
    """
    st.markdown(scrollable_container, unsafe_allow_html=True)

    st.audio(
        f"{convo_name}.mp3", start_time=int(st.session_state.start_t),
    )


def main():
    st.title("Candor Conversation Visualizer")

    if "start_t" not in st.session_state:
        st.session_state.start_t = 0
    if "end_t" not in st.session_state:
        st.session_state.end_t = 30

    filtered_data = load_data()
    backchannel_data = load_backchannels()

    col1, col2 = st.columns(2)

    with col1:
        feature1 = st.radio(
            "Select Feature 1:",
            (
                "pitch",
                "intensity",
                "backchannels",
                "surprisal",
                "entropy",
                "post_word_pause",
            ),
            index=0,
        )

    with col2:
        feature2 = st.radio(
            "Select Feature 2:",
            (
                "pitch",
                "intensity",
                "backchannels",
                "surprisal",
                "entropy",
                "post_word_pause",
            ),
            index=1,
        )
    feature1 = name_mapping.get(feature1, feature1)
    feature2 = name_mapping.get(feature2, feature2)

    context_len = int(
        st.selectbox(
            "Context Length (number of previous turns of context used to compute surprisal and entropy)",
            options=sorted(filtered_data.context_len.unique()),
        )
    )

    def convo_changed():
        st.session_state.start_t = 0
        st.session_state.end_t = 30

    convo_name = st.selectbox(
        "Conversation",
        options=sorted(filtered_data.transcript_name.unique()),
        on_change=convo_changed,
    )

    def time_changed():
        times_str = st.session_state.time_range_text_input
        start_t, end_t = times_str.split("-")
        start_t, end_t = float(start_t), float(end_t)
        st.session_state.start_t = start_t
        st.session_state.end_t = end_t

    times_str = st.text_input(
        "Time Range (seconds, dash-separated)",
        key="time_range_text_input",
        value=f"{st.session_state.start_t:.0f}-{st.session_state.end_t:.0f}",
        on_change=time_changed,
    )

    def backward():
        st.session_state.start_t -= 30
        st.session_state.end_t -= 30

    def forward():
        st.session_state.start_t += 30
        st.session_state.end_t += 30

    st.text(
        "Note: the audio recordings start when the first participant joins the call, so you may need to scrub forward to find the beginning of the conversation."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.button("<< 30s", on_click=backward)
    with col2:
        st.button("30s >>", on_click=forward)

    make_plot(
        feature1,
        feature2,
        filtered_data,
        context_len,
        convo_name,
        st.session_state.start_t,
        st.session_state.end_t,
        backchannel_data,
    )
    # make_plot("entropy", filtered_data)
    # make_plot("max_intensity", filtered_data)


if __name__ == "__main__":
    main()
