# %%
import base64
import io
import random
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

import dash
from dash import html, dcc, Output, Input, State
from dash.exceptions import PreventUpdate

import plotly.express as px
import plotly.graph_objects as go
import requests
import os
import gdown

h5_url = "https://drive.google.com/uc?export=download&id=1tTh4IUu_zsvoDnAeGp0aR2Fr1D34tOj2"
local_path = "data/processed_exp93to105.h5"

if not os.path.exists(local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print("Downloading .h5 file from Google Drive with gdown...")
    gdown.download(h5_url, local_path, quiet=False)
    print("Download complete.")
data = sc.read(local_path)

gif_url = "https://drive.google.com/uc?export=download&id=1N2Eu_zpINag9GjsKhFyudS2dx96inaL5"
gif_local_path = "assets/b-atlas-p2.gif"

# Download the GIF once if not present
if not os.path.exists(gif_local_path):
    os.makedirs(os.path.dirname(gif_local_path), exist_ok=True)
    print("Downloading GIF from Google Drive...")
    with requests.get(gif_url, stream=True) as r:
        r.raise_for_status()
        with open(gif_local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("GIF download complete.")

# Encode the image for use in the app
with open(gif_local_path, "rb") as gif_file:
    encoded_image = base64.b64encode(gif_file.read()).decode()
# %%


custom_colors = px.colors.qualitative.Set3 + px.colors.qualitative.Dark24


# Prepare UMAP coordinates
umap_df = pd.DataFrame(
    data.obsm['X_umap'], columns=['UMAP1', 'UMAP2'], index=data.obs_names
)



# Dropdown options from metadata
columns_to_show = ['Phenotype','Isotype','Isotype_subclass','Iso-Phenotype','Condition','Time']
metadata_options = [{'label': col, 'value': col} for col in columns_to_show]
gene_options = [{'label': g, 'value': g} for g in data.var_names]


# %%
metadata_notes = {
    'Phenotype': {
        'Blast': "Cycling blasts.",
        'Early ActB': "Activated B cells, primarily from King's Day 7 and Rene's Day 9 samples. Most resemble germinal center B cells, characterized by BCL6 expression.",
        'Late ActB': "IFN-stimulated activated B cells, generated from Day10 or Day13 dataset.",
        'PB': "Plasmablast.",
        'PB*': "Plasmablast primarily derived from Rene's MOD condition (6 days in Phase 1) Day9, and cells might exit low level contanmination.",
        'PBPC': "Plasmablasts and plasma cells. Based on UMAP positioning, cells toward the terminal end of the cluster are likely more differentiated plasma cells",
        'Pre-PB': "Cycling pre-plasmablasts.",
    },
    'Condition': {
        'King-D13': "King's 3 phase culture Day13. Details in Rene Cheng et al. Nature Communications, 2022",
        'King-D7': "King's 3 phase culture Day7 (end of phase 1). Details in Rene Cheng et al. Nature Communications, 2022",
        'RC*-D10': "Rene's MOD (*no CpG); CD40L/IL21(7 days)> IFNa/IL6/IL15(3 days)",
        'RC**-D9': "Rene's MOD (**shorten phase 1); CD40L/CpG/IL21(6 days)> IFNa/IL6/IL15(3 days).",
        'RC-D10': "Rene's 2 phase; CD40L/CpG/IL21(7 days)> IFNa/IL6/IL15(3 days)",
        'RC-D13': "Rene's 2 phase; CD40L/CpG/IL21(7 days)> IFNa/IL6/IL15(6 days). Details in Rene Cheng et al. Nature Communications, 2023",
   
    },
    'Isotype_subclass': {
    'M': "IgM is the first antibody produced during immune activation.",
    'G1': "IgG1 is the most abundant subclass in humans and in Rene's dataset.",
    'G2': "IgG2 responds to polysaccharides and weakly activates complement.",
    'G3': "IgG3 strongly activates complement and appears early in infection.",
    'G4': "IgG4 is the least abundant subclass; linked to tolerance and chronic exposure.",
    'A': "IgA protects mucosal surfaces like the gut and airways.",
    'E': "IgE mediates allergic responses; very few cells in Rene's dataset.",
    'Other': "Likely doublets or ambiguous isotype calls.",
    'lowIg': "Low immunoglobulin expressed cells, early-stage or undifferentiated activated B cells and blasts."
    },

   'Isotype': {
    'M': "IgM is the first antibody produced during immune activation.",
    'G': "IgG play a crucial role in long-term immunity.",
    'A': "IgA protects mucosal surfaces like the gut and airways.",
    'E': "IgE mediates allergic responses; very few cells in Rene's dataset.",
    'Other': "Likely doublets or ambiguous isotype calls.",
    'lowIg': "Low immunoglobulin expressed cells, early-stage or undifferentiated activated B cells and blasts."
    },
    'Time': {
    '10': "Day10 dataset",
    '13': "Day13 dataset",
    '7': "Day7 dataset",
    '9': "Day9 dataset",
    
    }
    
}

# %%


app = dash.Dash(__name__, title="B Cell Atlas")

app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=False),

        html.Div(
            id='page-container',
            children=[

                # ——— Landing Screen ———
                html.Div(
                    id='landing-container',
                    style={
                        'display': 'block',
                        'height': '100vh',
                        'overflow': 'hidden'
                    },
                    children=[

                        # full-bleed wrapper for the GIF
                        html.Div(
                            [
                                # GIF background
                                html.Img(
                                    src=f'data:image/png;base64,{encoded_image}',
                                    className='portal-gif'
                                ),

                                # Title overlay
                                html.Div(
                                    "AtlasView",
                                    style={
                                        'position': 'absolute',
                                        'top': '20px',
                                        'left': '30px',
                                        'zIndex': 10,
                                        'color': 'white',
                                        'fontSize': '18px',
                                        'backgroundColor': 'rgba(0,0,0,0.4)',
                                        'padding': '5px 10px',
                                        'borderRadius': '5px'
                                    }
                                ),

                                # Enter button
                                html.Div(
                                    dcc.Link(
                                        html.Button(
                                            "Single Cell Data Portal→",
                                            id='enter-text',
                                            className='enter-button',
                                            style={
                                                'backgroundColor': 'white',
                                                'color': 'black',
                                                'fontSize': '40px',
                                                'fontWeight': 'bold',
                                                'border': '2px solid black',
                                                'borderRadius': '8px',
                                                'padding': '10px 20px',
                                                'opacity': '0.6'
                                            }
                                        ),
                                        href='/main'
                                    ),
                                    style={
                                        'position': 'absolute',
                                        'top': '84%',
                                        'left': '64%',
                                        'transform': 'translate(-50%,-50%)',
                                        'zIndex': 10
                                    }
                                ),

                                # Landing footer
                                html.Footer(
                                    html.P(
                                        "© 2025 Rene Cheng Gibson",
                                        style={
                                            'textAlign': 'center',
                                            'color': 'grey',
                                            'fontSize': '14px',
                                            'margin': '0'
                                        }
                                    ),
                                    style={
                                        'position': 'absolute',
                                        'bottom': '0',
                                        'width': '100%',
                                        'zIndex': 10
                                    }
                                ),

                            ],
                            className='gif-wrapper'
                        ),

                    ]
                ),

                # ——— Main Dashboard ———
                html.Div(
                    id='main-content',
                    style={'display': 'none'},
                    children=[

                        html.H1([
                            "HUMAN B CELL ATLAS DASHBOARD",
                            html.Span("Beta", style={
                                        'fontSize': '16px',
                                        'color': 'white',
                                        'backgroundColor': '#d9534f',
                                        'borderRadius': '4px',
                                        'padding': '2px 8px',
                                        'marginLeft': '10px',
                                        'verticalAlign': 'middle',
                                        'fontWeight': '600'
                                    })
                                ],
                            style={'textAlign': 'center', 'fontSize': '32px'}
                        ),
                        html.H3(
                            "Single-cell transcriptomics viewer with metadata filtering and gene plots",
                            style={'textAlign': 'center', 'color': 'gray'}
                        ),
                        html.Div([
                        html.Span("Questions or feedback? "),
                        html.A("Email Rene", href="mailto:rene271828@gmail.com", style={'color': '#007BFF'})
                        ], style={
                            'textAlign': 'center',
                            'fontSize': '14px',
                            'color': 'gray',
                            'marginTop': '10px',
                            'marginBottom': '20px'
                            }),
                        html.Div([
                            html.Label("Color by metadata:", style={'fontSize': '18px'}),
                            dcc.Dropdown(
                                id='color-dropdown',
                                options=metadata_options,
                                value='Phenotype',
                                style={'fontSize': '16px'}
                            ),

                            html.Label("Filter by selected metadata values:", style={'fontSize': '18px'}),
                            dcc.Dropdown(
                                id='filter-value-dropdown',
                                multi=True,
                                style={'fontSize': '16px'}
                            ),

                            html.Label("Gene expression UMAP (enter gene name):", style={'fontSize': '18px'}),
                            dcc.Dropdown(
                                id='gene-input',
                                options=gene_options,
                                placeholder='Select or type a gene',
                                searchable=True,
                                style={'fontSize': '16px'}
                            ),

                            html.Div(id='gene-warning'),

                            html.Div([
                                html.Div(dcc.Graph(id='umap-plot'), className='fixed-graph',
                                        style={ 
                                                'padding': '10px',
                                                'boxSizing': 'border-box' }),
                                html.Div(dcc.Graph(id='gene-umap'), className='fixed-graph',
                                         style={
                                                'padding': '10px',
                                                'boxSizing': 'border-box' })
                            ], className='fixed-layout-container', ),
                                            
                            html.Div(id='filter-summary', style={'marginTop': '10px', 'fontSize': '14px', 'color': '#555'}),
          

                            html.Label(
                                "Expression heatmap: Compare expression across selected groups",
                                style={'fontSize': '18px'}
                            ),
                            dcc.Dropdown(
                                id='gene-multi-input',
                                options=gene_options,
                                multi=True,
                                style={'fontSize': '16px'}
                            ),
                            html.Div(dcc.Graph(id='dotplot'), style={'padding': '10px'})
                        ]),

                        html.Hr(),
                        html.Div([
                            html.H2("About", style={'fontSize': '20px'}),
                            html.P(
                                "This website was developed to explore and visualize B cell research data, supporting the James Lab, Rawlings Lab, and the broader B cell research community.",
                                style={'fontSize': '16px'}
                            ),
                            
                            html.P(
                                "Note: Immunoglobulin (IG) genes were removed from the dataset prior to analysis, "
                                "as their high expression levels can dominate clustering and obscure other transcriptional differences. "
                                "Cells were derived from PBMC-isolated B cells and differentiated in vitro.",

                                style={'fontSize': '14px', 'fontStyle': 'italic'}
                            ),
                            html.H4("References:"),
                            html.Ul([
                                html.Li(html.A(
                                    "Rene Cheng et al. Nature Communications, 2023",
                                    href="https://www.nature.com/articles/s41467-023-39367-8",
                                    target="_blank"
                                )),
                                html.Li(html.A(
                                    "Rene Cheng et al. Nature Communications, 2022",
                                    href="https://www.nature.com/articles/s41467-022-33787-8",
                                    target="_blank"
                                ))
                            ])
                        ], style={'margin': '30px'})

                    ]
                ),

            ],
            style={'minHeight': 'calc(100vh - 60px)'}
        ),

        # Footer for the dashboard pages
        html.Footer(
            html.P(
                "© 2025 Rene Cheng Gibson",
                style={'textAlign': 'center', 'color': 'gray', 'fontSize': '14px', 'margin': '10px 0'}
            ),
            id='dashboard-footer',
            style={'display': 'none', 'width': '100%', 'textAlign': 'center'}
        )

    ],
    className='app-container'
)


# %%
@app.callback(
    Output('landing-container', 'style'),
    Output('main-content', 'style'),
    Output('dashboard-footer', 'style'),
    Input('url', 'pathname')
)
def route(pathname):
    if pathname == '/main':
        return (
            {'display': 'none'},
            {'display': 'block', 'padding': '20px'},
            {'display': 'block'}  # show footer
        )
    return (
        {'display': 'block'},
        {'display': 'none'},
        {'display': 'none'}  # hide footer
    )

# Callback to update plot
@app.callback(
    Output('filter-value-dropdown', 'options'),
    Output('filter-value-dropdown', 'value'),
    Input('color-dropdown', 'value')
)
def update_filter_values(selected_meta):
    values = sorted(data.obs[selected_meta].dropna().astype(str).unique())
    options = [{'label': v, 'value': v} for v in values]
    return options, values  # default: select all values

@app.callback(
    Output('umap-plot', 'figure'),
    Input('color-dropdown', 'value'),
    Input('filter-value-dropdown', 'value')
)

def update_umap_plot(meta_col, selected_values):
    df = umap_df.copy()
    df[meta_col] = data.obs[meta_col].astype(str)
    selected_values = selected_values or []
    if selected_values:
        df = df[df[meta_col].isin(selected_values)]

    # Shuffle rows (to randomize draw order)
    df = df.sample(frac=1).reset_index(drop=True)

    # Assign a color to each group value
    unique_vals = df[meta_col].unique()
    color_map = {val: custom_colors[i % len(custom_colors)] for i, val in enumerate(unique_vals)}
    df['color'] = df[meta_col].map(color_map)

    # Build ONE trace only
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=df['UMAP1'],
        y=df['UMAP2'],
        mode='markers',
        marker=dict(
            size=4,
            color=df['color'],  # ✅ point-level color control
            opacity=0.7,
            line=dict(width=0)
        ),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Add dummy traces just for the legend
    for val in sorted(unique_vals):
        fig.add_trace(go.Scattergl(
            x=[None], y=[None],
            mode='markers',
            name=str(val),
            marker=dict(size=6, color=color_map[val]),
            showlegend=True
        ))

    fig.update_layout(
        width=600,
        height=550,
        margin=dict(l=0, r=110, t=40, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        uirevision=True,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        legend=dict(
            font=dict(size=14),
            title_font=dict(size=16),
            itemsizing='constant',
            x=1.02, xanchor='left', y=1
        ),
        shapes=[dict(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="gray", width=1)
        )]
    )

    return fig 

  

# --- Gene expression UMAP callback ---
@app.callback(
    Output('gene-umap', 'figure'),
    Output('gene-warning', 'children'),
    Input('gene-input', 'value'),
    Input('color-dropdown', 'value'),
    Input('filter-value-dropdown', 'value')
)
def update_gene_plot(gene, meta_col, selected_values):
    df = umap_df.copy()
    df[meta_col] = data.obs[meta_col].astype(str)
    selected_values = selected_values or []
    if selected_values:
        df = df[df[meta_col].isin(selected_values)]

    idx = df.index
    fig = go.Figure()
    

    if gene and gene in data.var_names:
        expression = data[idx, gene].X.toarray().flatten()
        df['expression'] = expression

        # Background layer: all cells in grey
        fig.add_trace(go.Scatter(
            x=df['UMAP1'], y=df['UMAP2'],
            mode='markers',
            marker=dict(size=4, color='whitesmoke'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Foreground layer: cells with expression > 0
        df_expr = df[df['expression'] > 0]
        fig.add_trace(go.Scatter(
            x=df_expr['UMAP1'], y=df_expr['UMAP2'],
            mode='markers',
            marker=dict(
                size=4,
                color=df_expr['expression'],
                colorscale='blues',
                colorbar=dict(title=None, xpad=10),
                showscale=True
            ),
            name=gene,
            showlegend=False, 
            hovertemplate='Expression: %{marker.color:.2f}<extra></extra>'
        ))

        fig.update_layout(
            width=600,
            height=550,
            margin=dict(l=0, r=110, t=40, b=40),
            title=f'{gene} expression', 
            uirevision=True, 
            plot_bgcolor='white',
            paper_bgcolor='white',
            shapes=[dict(
                type="rect",
                xref="paper", yref="paper",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color="gray", width=1)
            )],
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            
        )
        return fig, ""

    else:
        # If no gene is selected or gene is invalid, show just grey UMAP + message
        fig.add_trace(go.Scatter(
            x=df['UMAP1'], y=df['UMAP2'],
            mode='markers',
            marker=dict(size=5, color='whitesmoke',),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_annotation(
            text="Awaiting gene selection",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            width=600,
            height=550,
            margin=dict(l=0, r=110, t=40, b=40),
            uirevision=True, 
            plot_bgcolor='white',
            paper_bgcolor='white',
            shapes=[dict(
                type="rect",
                xref="paper", yref="paper",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color="gray", width=1)
            )],
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            
        )
        return fig, f"Gene '{gene}' not found." if gene else ""

@app.callback(
    Output('filter-summary', 'children'),
    Input('filter-value-dropdown', 'value'),
    State('color-dropdown', 'value')
)
def update_filter_summary(selected_values, meta_col):
    if not selected_values:
        return f"No filters applied to {meta_col}. Showing all cells."

    # Special note for composite metadata
    if meta_col in ['Iso-Phenotype']:
        return html.Div([
            html.H5("Selected Iso-Phenotype groups:", style={'marginBottom': '5px'}),
            html.P("Combined phenotype and isotype classification capturing both differentiation state and antibody class."),
        ])

    meta_dict = metadata_notes.get(meta_col, {})
    notes = []
    for val in selected_values:
        desc = meta_dict.get(val, "No description available.")
        notes.append(html.Li(f"{val}: {desc}"))

    return html.Div([
        html.H5(f"Selected {meta_col} notes:", style={'marginBottom': '5px'}),
        html.Ul(notes)
    ])




@app.callback(
    Output('dotplot', 'figure'),
    Input('gene-multi-input', 'value'),
    Input('color-dropdown', 'value'),
    Input('filter-value-dropdown', 'value')
)
def update_dotplot(gene_list, meta_col, selected_values):
    if not gene_list:
        fig = go.Figure()
        fig.add_annotation(
            text="Awaiting gene selection",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        return fig

    if any(g not in data.var_names for g in gene_list):
        return go.Figure(data=[])

    obs = data.obs[meta_col].astype(str)
    filtered_idx = obs[obs.isin(selected_values)].index
    filtered_data = data[filtered_idx]

    ordered_categories = {
        'Phenotype': [
            'Early ActB', 'Late ActB', 'Blast',
            'Pre-PB', 'PB*', 'PB', 'PBPC'
        ],
        'Iso-Phenotype': [
            'Early ActB', 'Late ActB', 'Blast',
            'M prePB','G prePB','A prePB', 
            'M PB*', 'G PB*','A PB*',
            'M PB', 'G PB','A PB',
            'M PBPC', 'G PBPC','A PBPC',
            'Other'

        ]
    }
    if meta_col in ordered_categories:
        filtered_data.obs[meta_col] = pd.Categorical(
            filtered_data.obs[meta_col].astype(str),
            categories=ordered_categories[meta_col],
            ordered=True
        )
    else:
        filtered_data.obs[meta_col] = filtered_data.obs[meta_col].astype(str)

    expr_matrix = filtered_data[:, gene_list].X
    if not isinstance(expr_matrix, np.ndarray):
        expr_matrix = expr_matrix.toarray()
    expr_df = pd.DataFrame(expr_matrix, index=filtered_data.obs_names, columns=gene_list)
    expr_df[meta_col] = filtered_data.obs[meta_col].values

    avg_expr = expr_df.groupby(meta_col).mean().reset_index().melt(id_vars=meta_col, var_name="gene", value_name="avg_expr")
    pct_expr = expr_df[gene_list].gt(0).groupby(expr_df[meta_col]).mean().reset_index().melt(id_vars=meta_col, var_name="gene", value_name="pct_expr")
    plot_df = avg_expr.merge(pct_expr, on=[meta_col, 'gene'])

    # 🧼 Drop rows with missing data (e.g., gene not expressed in a group)
    plot_df = plot_df.dropna(subset=["avg_expr", "pct_expr"])
    
    fig = px.scatter(
        plot_df, x=meta_col, y="gene", size="pct_expr", color="avg_expr",
        color_continuous_scale="Blues", size_max=20,
        symbol_sequence=["square"] * len(plot_df)
    )
    fig.update_traces(marker=dict(line=dict(width=2, color="black")))
    fig.update_layout(
        width=120 + 50 * plot_df[meta_col].nunique(),
        height=120 + 44 * len(gene_list),
        title=f"Expression of {len(gene_list)} gene(s) grouped by {meta_col}",
        xaxis_title="",
        yaxis_title="Gene",
        margin=dict(t=40, l=0, r=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig
     


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8053)



