``` bash
$ cd [repository-name]
$ git checkout --orphan gh-pages
$ git rm -rf .
$ Download contents
$ git commit -a -m "First commit"
$ bundle install
$ bundle exec jekyll serve -w
# edit `_config.yml` so that `baseurl` is `"/repository-name"`, commit, and push to Github:
$ git push origin gh-pages
# Check https://[username].github.io/[repository-name]
```
