% This script requires https://github.com/kwikteam/npy-matlab
clear; clc;
ieee_configure();
addpath(genpath("environments"));
addpath(genpath("outputs"));
[X1, X2] = meshgrid(0: 0.01: 1 - 0.01, 0: 0.01: 1 - 0.01);

% Visualize environment
raw_data = readNPY('../data/volcano/preprocessed/helens.npy');
x_train = readmatrix('../data/volcano/preprocessed/x_train.csv');
y_train = readmatrix('../data/volcano/preprocessed/y_train.csv');
plot_map(X1, X2, raw_data, 'env', x_train, y_train);

% Visualize RBF
load ./outputs/rbf.mat;
plot_map(X1, X2, mean, 'rbf_mean');
plot_map(X1, X2, 2 * std, 'rbf_std');
plot_map(X1, X2, error, 'rbf_error');

% Visualize AK
load ./outputs/ak.mat;
plot_map(X1, X2, mean, 'ak_mean');
plot_map(X1, X2, 2 * std, 'ak_std');
plot_map(X1, X2, error, 'ak_error');


function plot_map(X1, X2, map, name, x_train, y_train)
    surf(X1, X2, reshape(map, 100, 100));
    hold on;
    if nargin > 4
        scatter3(x_train(:, 1), ...
            x_train(:, 2), ...
            y_train, ...
            'Filled', ...
            'MarkerFaceColor', 'k');
    end
    colormap('jet');
    colorbar("north");
    if contains(name, "error")
        caxis([0, 900]);
    elseif contains(name, "env") || contains(name, "mean")
        caxis([2.7399e+03, 8.2467e+03]);
    end
    view(-70, 60);
    ax = gca;
    ax.FontSize = 30;
    shading interp;
    camlight;
    lighting phong;
    hold off;
    ax = gca;
    ax.ZAxis.Visible = 'off';
    ax.XAxis.Visible = 'off';
    ax.YAxis.Visible = 'off';
    grid off
    print(['./figures/' name], '-dpng');
    close;
end

function ieee_configure()
    width = 516;
    fraction = 1;
    fig_width_pt = width * fraction;
    inches_per_pt = 1 / 72.27;
    golden_ratio = (sqrt(5) - 1) / 2;
    width = fig_width_pt * inches_per_pt;
    hight = width * golden_ratio;
    %% figure margins
    top = 0;  % normalized top margin
    bottom = 0;	% normalized bottom margin
    left = 0.1;	% normalized left margin
    right = 0.1;  % normalized right margin
    %% set default figure configurations
    set(0,'defaultFigureUnits','inches');
    set(0,'defaultFigurePosition',[0 0 width hight]);
    set(0,'defaultAxesFontName','Times New Roman');
    set(0,'defaultAxesFontSize',8);
    set(0,'defaultTextFontName','Times New Roman');
    set(0,'defaultTextFontSize',10);
    set(0,'defaultLegendFontName','Times New Roman');
    set(0,'defaultLegendFontSize',8);
    set(0,'defaultAxesUnits','normalized');
    set(0,'defaultAxesPosition',...
        [left/width ...
        bottom/hight ...
        (width-left-right)/width ...
        (hight-bottom-top)/hight]);
end