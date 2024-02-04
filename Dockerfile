FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Install basic dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update -y && \
	apt install -y --no-install-recommends \
	curl \
	git && \
	rm -rf /var/lib/apt/lists/*

# Reset the conda environment back to the beginning
RUN conda install --name base --revision 0 -y && \
	conda clean --all -y

# Install python 3.11 and cuda deps for pytorch (as pytorch do)
RUN conda install -y -c pytorch -c nvidia python=3.11 pytorch-cuda=11.8 && \
	conda install -y -c pytorch pytorch && \
	# Install PDM
	conda install -y -c conda-forge pdm && \
	# Install GitHub CLI
	conda install -y -c conda-forge gh && \
	conda clean --all -y

ENTRYPOINT [ "/bin/bash" ]
