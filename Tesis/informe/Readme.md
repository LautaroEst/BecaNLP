# Algunas cosas para agregar al documento

---

## Figuras y subfiguras:

\begin{figure}[h]
	\centering
	\includegraphics[width=0.5\textwidth]{fig1}
	\caption{Mi figura}
	\label{fig:mi figura}
\end{figure}


\begin{figure}[h]
	\centering
	\begin{subfigure}[b]{0.3\textwidth}
		\centering
		\includegraphics[width=\textwidth]{fig1}
		\caption{Mi figura}
		\label{fig:mi figura}
	\end{subfigure}
	\hfill
	\begin{subfigure}[b]{0.3\textwidth}
		\centering
		\includegraphics[width=\textwidth]{fig1}
		\caption{Mi figura}
		\label{fig:mi figura}
	\end{subfigure}
	\hfill
	\begin{subfigure}[b]{0.3\textwidth}
		\centering
		\includegraphics[width=\textwidth]{fig1}
		\caption{Mi figura}
		\label{fig:mi figura}
	\end{subfigure}
	\caption{Las tres figuras}
	\label{fig:tres figuras juntas}
\end{figure}


## Tablas y subtablas:

\begin{table}[h]
\centering
\begin{tabular}{r | c | l}
A & B & C \\
\hline
1 & 2 & 3 \\
4 & 5 & 6 
\end{tabular}
\caption{Tabla básica}
\label{tab:basic table}
\end{table}


\begin{table}[h]
	\begin{subtable}[b]{0.45\textwidth}
		\centering
		\begin{tabular}{r | c | l}
		A & B & C \\
		\hline
		1 & 2 & 3 \\
		4 & 5 & 6 
		\end{tabular}
		\caption{Mi figura}
		\label{fig:mi figura}
	\end{subtable}
	\hfill
	\begin{subtable}[b]{0.45\textwidth}
		\centering
		\begin{tabular}{r | c | l}
		A & B & C \\
		\hline
		1 & 2 & 3 \\
		4 & 5 & 6 
		\end{tabular}
		\caption{Mi figura}
		\label{fig:mi figura}
	\end{subtable}
	\caption{Las dos tablas}
	\label{tab:dos figuras juntas}
\end{table}











