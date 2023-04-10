# Contributing

**IMPORTANT: The Readme file in this repository is generated. Please DO NOT edit it directly!**

Structure of this repo:

- `README.md`: The main, auto-built index file.
- `data`: The data folder containing all indexed resources in yaml format.
- `scripts`: Scripts for building the index file.

How to add a new resource:

1. Set up environment:
    ```
    pip install -r scripts/requirements.txt
    ```

2. Select the most fitting file for the new resource in `data` (or add a new file).

3. Add a new list item for the new resource under `items` in the selected yaml file.
   The template below shows all supported keys:
    ```yaml
    title: "Paper title"
    paper:
      acl_id: ""
      arxiv_id: ""
      doi: ""
      semantic_scholar_id: ""
      pdf: https://...
    code: https://github.com/...
    tags: ["Tag1", "Tag2"]
    ```

4. Run the build script:
    ```
    python scripts/build.py
    ```

5. Check if the output in `README.md` looks good.

6. Commit your changes and open a pull request!
